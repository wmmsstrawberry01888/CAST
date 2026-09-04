import torch
import torch.nn as nn
import torch.nn.functional as F
# this efficient implementation comes from https://github.com/xptree/DeepInf/


class GMLP(nn.Module):
    def __init__(self, feature_size):
        super(GMLP, self).__init__()
        self.feature_size = feature_size
        self.e = nn.Linear(self.feature_size, self.feature_size)
        self.r = nn.Linear(self.feature_size, self.feature_size)
        self.l = nn.Linear(self.feature_size, self.feature_size)
        self.grelu = nn.Tanh()

    def forward(self, x):
        x1 = self.e(x)
        x2 = self.r(x)
        return self.l(self.grelu(x1) * x2 + x)

class BatchMultiHeadGraphAttention(nn.Module):
    def __init__(self, n_head, f_in, f_out, num_v, attn_dropout, bias=True):
        super(BatchMultiHeadGraphAttention, self).__init__()
        self.n_head = n_head
        self.f_in = f_in
        self.f_out = f_out
        self.w = nn.Parameter(torch.Tensor(n_head, f_in, f_out))
        self.a_src = nn.Parameter(torch.Tensor(n_head, f_out, num_v))
        self.a_dst = nn.Parameter(torch.Tensor(n_head, f_out, num_v))

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(attn_dropout)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(f_out))
            nn.init.constant_(self.bias, 0)
        else:
            self.register_parameter("bias", None)

        nn.init.xavier_uniform_(self.w, gain=1.414)
        nn.init.xavier_uniform_(self.a_src, gain=1.414)
        nn.init.xavier_uniform_(self.a_dst, gain=1.414)

    def forward(self, h):
        bs, n = h.size()[:2]
        self.w = self.w.to(h.device)
        self.a_src = self.a_src.to(h.device)
        self.a_dst = self.a_dst.to(h.device)
        h_prime = torch.matmul(h.unsqueeze(1), self.w)
        attn_src = torch.matmul(h_prime, self.a_src)
        attn_dst = torch.matmul(h_prime, self.a_dst)
        attn = attn_src.expand(-1, -1, -1, n) + attn_dst.expand(-1, -1, -1, n).permute(
            0, 1, 3, 2
        )
        attn = attn_src + attn_dst
        attn = self.leaky_relu(attn)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.matmul(attn, h_prime)
        if self.bias is not None:
            return output + self.bias, attn
        else:
            return output, attn

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.n_head)
            + " -> "
            + str(self.f_in)
            + " -> "
            + str(self.f_out)
            + ")"
        )


class GAT(nn.Module):
    def __init__(self, n_units, n_heads, num_v, dropout=0.2, alpha=0.2):
        super(GAT, self).__init__()
        self.n_layer = len(n_units) - 1
        self.dropout = dropout
        self.layer_stack = nn.ModuleList()

        for i in range(self.n_layer):
            f_in = n_units[i] * n_heads[i - 1] if i else n_units[i]
            self.layer_stack.append(
                BatchMultiHeadGraphAttention(
                    n_heads[i], f_in=f_in, f_out=n_units[i + 1], num_v=num_v, attn_dropout=dropout
                )
            )

        self.norm_list = nn.ModuleList()
        for i in range(self.n_layer):
            f = n_units[i] * n_heads[i - 1] if i else n_units[i]
            self.norm_list.append(torch.nn.InstanceNorm1d(f))

    def forward(self, x):
        bs, n = x.size()[:2]
        for i, gat_layer in enumerate(self.layer_stack):
            x = self.norm_list[i](x.permute(0, 2, 1)).permute(0, 2, 1)
            x, attn = gat_layer(x)
            if i + 1 == self.n_layer:
                x = x.squeeze(dim=1)
            else:
                x = F.gelu(x.transpose(1, 2).contiguous().view(bs, n, -1))
                x = F.dropout(x, self.dropout, training=self.training)
        else:
            return x


class GATEncoder(nn.Module):
    def __init__(self, n_units, n_heads, num_v, dropout=0.2, alpha=0.2):
        super(GATEncoder, self).__init__()
        self.gat = GAT(n_units, n_heads, num_v, dropout, alpha)
        self.feature_out = n_units[-1]

    def forward(self, x):
        bs, seq_len, n = x.size()[:3]
        final_feature = torch.Tensor(bs, seq_len, n, self.feature_out).to(x.device)
        for i in range(seq_len):
            final_feature[:, i] = self.gat(x[:, i])
        return final_feature

