import torch
import torch.nn as nn
from model.layer import GATEncoder
from model.diffusion_model import DiffSTG
from model.attention import TemporalEncoder
from model.resnet import ResNet
from model.attention import PositionalEncoding
from model.layer import GMLP
from layers.network import Network


class Cast(nn.Module):
    """CAST: Custom Attention-based Spatio-Temporal model.

    Parameters
    ----------
    config : easydict
        Model configuration (T_h, T_p, F, V, etc.)
    n_units : list[int]
        Channel dimensions per GAT layer, e.g. [F, 16, 16].
    n_heads : list[int]
        Number of attention heads per GAT layer, e.g. [4, 1].
    num_v : int
        Number of spatial nodes (V).
    use_gat : bool
        Enable GAT spatial encoder (FPG).
    use_temporal : bool
        Enable TemporalEncoder (STDC).
    use_spatio : bool
        Enable ResNet spatial encoder.
    use_network : bool
        Enable Network fusion module.
    use_diffusion : bool
        Enable conditional denoising diffusion generator (CDG).
    """

    def __init__(self, config, n_units, n_heads, num_v, dropout=0.2, alpha=0.2,
                 use_gat=True, use_temporal=True, use_spatio=True,
                 use_network=True, use_diffusion=False):
        super(Cast, self).__init__()
        self.is_use_diffusion_model = use_diffusion
        self.is_use_gat_encoder = use_gat
        self.is_use_temporal_encoder = use_temporal
        self.is_use_spatio_encoder = use_spatio
        self.is_use_net_work = use_network

        if self.is_use_gat_encoder:
            self.gat = GATEncoder(n_units, n_heads, num_v, dropout=0.2, alpha=0.2)
            self.gat_proj = nn.Linear(n_units[0], n_units[-1])
            self.gat_cat = nn.Linear(n_units[-1] * 2, n_units[-1])
        else:
            self.gat = nn.Linear(n_units[0], n_units[-1])
        self.hidden_dim = n_units[-1]

        if self.is_use_temporal_encoder:
            self.t = TemporalEncoder(
                d_model=self.hidden_dim * num_v,
                out_dim=self.hidden_dim * num_v,
                n_heads=8,
                atten_layer=6
            )
        else:
            self.t = nn.Linear(self.hidden_dim * num_v, self.hidden_dim * num_v)

        if self.is_use_spatio_encoder:
            self.s = ResNet(config.T_h, config.T_h)
        else:
            self.s = nn.Linear(self.hidden_dim * num_v, self.hidden_dim * num_v)

        self.fusion = nn.Linear(self.hidden_dim, self.hidden_dim)

        if self.is_use_net_work:
            self.net = Network(config.T_h, config.T_h, config.T_h, 3, 'end')
        else:
            self.net = nn.Linear(self.hidden_dim * num_v * 2, self.hidden_dim * num_v)
        self.PE = PositionalEncoding(max_len=config.T_h, d_model=self.hidden_dim * num_v)

        self.layer_norm = nn.LayerNorm(self.hidden_dim * num_v)

        self.layer_norm2 = nn.LayerNorm(self.hidden_dim)


        F_orig = config.F
        inner_F = self.hidden_dim
        if self.is_use_diffusion_model:
            import copy
            diff_cfg = copy.deepcopy(config)
            diff_cfg.F = inner_F
            self.diffusion = DiffSTG(diff_cfg)
        else:
            self.diffusion = nn.Linear(inner_F, inner_F)
        self.decoder = GMLP(inner_F)
        self.final_proj = nn.Linear(inner_F, F_orig)
        self.tanh = nn.GELU()

        # Cached context for the CDG denoising loss (populated by forward()).
        self._hist_gat_enc = None
        self._cdg_cond = None

    def _encode_gat(self, x):
        """Shared FPG graph encoder over flight-parameter types.

        Maps a raw sequence (B, T, V, F) to its latent representation
        (B, T, V, hidden_dim). Used both for the history (inside forward) and,
        during training, for the ground-truth future to build the clean target
        y_0 of the CDG denoising loss.
        """
        if self.is_use_gat_encoder:
            enc = self.gat(x)
            proj = self.gat_proj(x)
            enc = self.gat_cat(torch.cat((enc, proj), dim=-1))
        else:
            enc = self.gat(x)
        return enc

    def forward(self, x, pos_w, pos_d):
        self.encoder = self._encode_gat(x)
        # Cache the history latent + diffusion condition so that cdg_loss() can
        # reuse the exact encoded flight context Z_{1:T} of this forward pass.
        self._hist_gat_enc = self.encoder
        self._cdg_cond = None
        y1, y2, y3, y4 = self.encoder.shape
        self.trans_enc = self.encoder.view(y1, y2, -1)

        self.trans_enc = self.layer_norm(self.trans_enc)

        self.trans_enc = self.PE(self.trans_enc)

        self.t_hidden = self.t(self.trans_enc)
        self.s_hidden = self.s(self.trans_enc)

        if self.is_use_net_work:
            self.st_hidden = self.net(self.s_hidden, self.t_hidden)
        else:
            fusion_features = torch.cat((self.s_hidden, self.t_hidden), dim=-1)
            self.st_hidden = self.net(fusion_features)
        self.st_hidden = self.st_hidden.view(y1, y2, y3, -1)

        self.hidden = self.fusion(self.st_hidden)

        self.hidden = self.layer_norm2(self.hidden)

        if self.is_use_diffusion_model:
            x_masked = torch.cat((self.hidden, torch.zeros_like(self.hidden)), dim=1).to(
                self.hidden.device)  # (B, T, V, F)
            x_masked = x_masked.transpose(1, 3)  # (B, F, V, T)

            n_samples = 2

            x_masked = self.tanh(x_masked)  # [8, 3, 5, 24]
            # Cache the encoded flight context Z_{1:T} used as the diffusion
            # condition, so cdg_loss() trains eps_theta on the same context.
            self._cdg_cond = (x_masked, pos_w, pos_d)
            self.diff_hidden = self.diffusion((x_masked, pos_w, pos_d), n_samples)
            self.diff_hidden = torch.nan_to_num(self.diff_hidden, nan=-1)
            self.diff_hidden = self.diff_hidden.transpose(2, 4)
            self.diff_hidden = self.diff_hidden[:, 0]
        else:
            self.diff_hidden = self.diffusion(self.hidden)
        self.diff_hidden = self.tanh(self.diff_hidden)
        self.diff_hidden = self.decoder(self.diff_hidden)
        self.diff_hidden = self.final_proj(self.diff_hidden)
        return self.diff_hidden

    def cdg_loss(self, future):
        """Conditional denoising loss L_cdg of the CDG (paper Eq. L_cdg).

        The clean target y_0 is the ground-truth future flight-parameter
        trajectory encoded by the shared FPG graph encoder into the same latent
        space in which the CDG operates. Conditioned on the encoded flight
        context Z_{1:T} cached during the preceding forward() call, this returns
        E_{t, y_0, eps}[|| eps - eps_theta(sqrt(a_bar_t) y_0 + sqrt(1 - a_bar_t)
        eps, t, Z) ||^2], i.e. the standard noise-prediction objective.

        Returns a zero-valued scalar when the diffusion generator is disabled
        (ablations O-Net without CDG), so callers can add it unconditionally.

        Parameters
        ----------
        future : torch.Tensor
            Ground-truth future trajectory (B, T_p, V, F), same layout as the
            forward() input.
        """
        if not self.is_use_diffusion_model or self._cdg_cond is None:
            ref = self._hist_gat_enc if self._hist_gat_enc is not None else future
            return torch.zeros((), device=ref.device, dtype=torch.float32)

        # Encode the ground-truth future with the shared FPG encoder, then lay
        # it out exactly like the diffusion condition x_masked (first half =
        # encoded signal, second half = zeros; channels last -> F axis).
        fut_enc = self._encode_gat(future)               # (B, T, V, hidden)
        y0 = torch.cat((fut_enc, torch.zeros_like(fut_enc)), dim=1)
        y0 = y0.transpose(1, 3)                          # (B, hidden, V, 2T)
        return self.diffusion.loss(y0, self._cdg_cond)