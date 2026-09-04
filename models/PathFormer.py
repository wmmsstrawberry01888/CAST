import torch.nn as nn
from layers.AMS import AMS
from layers.revin import RevIN


class Model(nn.Module):
    def __init__(self, layer_nums, num_nodes, pred_len, seq_len, k, num_experts_list, patch_size_list,
                 d_model, d_ff, residual_connection, revin, device, batch_norm):
        super(Model, self).__init__()
        self.layer_nums = layer_nums  # number of pathway layers
        self.num_nodes = num_nodes
        self.pre_len = pred_len
        self.seq_len = seq_len
        self.k = k
        self.num_experts_list = num_experts_list
        self.patch_size_list = patch_size_list
        self.d_model = d_model
        self.d_ff = d_ff
        self.residual_connection = residual_connection
        self.revin = revin
        if self.revin:
            self.revin_layer = RevIN(num_features=num_nodes, affine=False, subtract_last=False)

        self.start_fc = nn.Linear(in_features=1, out_features=self.d_model)
        self.AMS_lists = nn.ModuleList()
        self.device = device
        self.batch_norm = batch_norm

        for num in range(self.layer_nums):
            self.AMS_lists.append(
                AMS(self.seq_len, self.seq_len, self.num_experts_list[num], self.device, k=self.k,
                    num_nodes=self.num_nodes, patch_size=self.patch_size_list[num], noisy_gating=True,
                    d_model=self.d_model, d_ff=self.d_ff, layer_number=num + 1, residual_connection=self.residual_connection, batch_norm=self.batch_norm))
        self.projections = nn.Sequential(
            nn.Linear(self.seq_len * self.d_model, self.pre_len)
        )

    def forward(self, x):

        balance_loss = 0
        # norm
        if self.revin:
            x = self.revin_layer(x, 'norm')
        out = self.start_fc(x.unsqueeze(-1))


        batch_size = x.shape[0]

        for layer in self.AMS_lists:
            out, aux_loss = layer(out)
            balance_loss += aux_loss

        out = out.permute(0,2,1,3).reshape(batch_size, self.num_nodes, -1)
        out = self.projections(out).transpose(2, 1)

        # denorm
        if self.revin:
            out = self.revin_layer(out, 'denorm')

        return out, balance_loss


