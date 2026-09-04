import torch
import torch.nn as nn


class encoder(nn.Module):
    def __init__(self, seq_len, hidden_size, num_experts):
        super(encoder, self).__init__()
        input_size = seq_len
        num_experts = num_experts
        encoder_hidden_size = hidden_size

        self.distribution_fit = nn.Sequential(nn.Linear(input_size, encoder_hidden_size, bias=False),
                                              nn.ReLU(),
                                              nn.Linear(encoder_hidden_size, num_experts, bias=False))

    def forward(self, x):
        mean = torch.mean(x, dim=-1)
        out = self.distribution_fit(mean)
        return out
