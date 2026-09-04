import torch
import torch.nn as nn
from model.layer import GATEncoder
from model.diffusion_model import DiffSTG
from model.attention import TemporalEncoder
from model.resnet import ResNet
from model.attention import PositionalEncoding
from model.layer import GMLP


class STDiff(nn.Module):
    def __init__(self, config, n_units, n_heads, num_v, dropout=0.2, alpha=0.2):
        super(STDiff, self).__init__()

        self.gat = GATEncoder(n_units, n_heads, num_v, dropout=0.2, alpha=0.2)
        self.hidden_dim = n_units[-1]

        self.t = TemporalEncoder(
            d_model=self.hidden_dim * num_v,
            out_dim=self.hidden_dim * num_v,
            n_heads=8,
            atten_layer=6
        )

        self.s = ResNet(config.T_h, config.T_h)

        self.fusion = nn.Linear(self.hidden_dim * 2, config.F)

        self.cross_attention1 = nn.MultiheadAttention(
            embed_dim=self.hidden_dim * num_v,
            num_heads=8,
            batch_first=True
        )

        self.cross_attention2 = nn.MultiheadAttention(
            embed_dim=self.hidden_dim * num_v,
            num_heads=8,
            batch_first=True
        )

        self.PE = PositionalEncoding(max_len=20, d_model=self.hidden_dim * num_v)

        self.layer_norm = nn.LayerNorm(self.hidden_dim * num_v)

        self.layer_norm2 = nn.LayerNorm(config.F)

        self.is_use_diffusion_model = True
        # self.is_use_diffusion_model = False

        if self.is_use_diffusion_model:
            self.diffusion = DiffSTG(config)
        else:
            self.diffusion = nn.Linear(config.F, config.F)
        self.decoder = GMLP(config.F)
        self.final_proj = nn.Linear(config.F, config.F)
        self.relu = nn.ReLU()
        self.tanh = nn.GELU()

    def forward(self, x, pos_w, pos_d):
        self.encoder = self.gat(x)
        y1, y2, y3, y4 = self.encoder.shape
        self.trans_enc = self.encoder.view(y1, y2, -1)

        # self.trans_enc = self.tanh(self.trans_enc)

        self.trans_enc = self.layer_norm(self.trans_enc)

        self.trans_enc = self.PE(self.trans_enc)


        # self.t_hidden = self.t(self.trans_enc).view(y1, y2, y3, -1)
        # self.s_hidden = self.s(self.trans_enc).view(y1, y2, y3, -1)

        self.t_hidden = self.t(self.trans_enc)
        self.s_hidden = self.s(self.trans_enc)


        # self.st_hidden = torch.cat((self.t_hidden, self.s_hidden), dim=-1)

        self.t_hidden = self.PE(self.t_hidden)
        self.s_hidden = self.PE(self.s_hidden)

        # self.s_cross_hidden, _ = self.cross_attention1(self.t_hidden, self.s_hidden, self.s_hidden)
        # self.t_cross_hidden, _ = self.cross_attention2(self.s_hidden, self.t_hidden, self.t_hidden)

        self.t_cross_hidden, _ = self.cross_attention1(self.t_hidden, self.trans_enc, self.trans_enc)
        self.s_cross_hidden, _ = self.cross_attention2(self.s_hidden, self.trans_enc, self.trans_enc)

        self.s_cross_hidden = self.s_cross_hidden.view(y1, y2, y3, -1)
        self.t_cross_hidden = self.t_cross_hidden.view(y1, y2, y3, -1)

        # self.st_hidden = torch.cat((self.t_cross_hidden, self.s_cross_hidden,  self.encoder), dim=-1)

        self.st_hidden = torch.cat((self.t_cross_hidden, self.s_cross_hidden), dim=-1)

        self.hidden = self.fusion(self.st_hidden)
        self.hidden = self.layer_norm2(self.hidden)

        if self.is_use_diffusion_model:
            x_masked = torch.cat((self.hidden, torch.zeros_like(self.hidden)), dim=1).to(
                self.hidden.device)  # (B, T, V, F)
            x_masked = x_masked.transpose(1, 3)  # (B, F, V, T)

            n_samples = 2

            x_masked = self.tanh(x_masked)  # [8, 3, 5, 24]
            self.diff_hidden = self.diffusion((x_masked, pos_w, pos_d), n_samples)
            self.diff_hidden = torch.nan_to_num(self.diff_hidden, nan=-1)
            self.diff_hidden = self.diff_hidden.transpose(2, 4)
            self.diff_hidden = self.diff_hidden[:, 0]
        else:
            self.diff_hidden = self.diffusion(self.hidden)
            # self.diff_hidden = self.diff_hidden.transpose(1, 3)
        self.diff_hidden = self.tanh(self.diff_hidden)
        self.diff_hidden = self.decoder(self.diff_hidden)
        self.diff_hidden = self.final_proj(self.diff_hidden)
        return self.diff_hidden