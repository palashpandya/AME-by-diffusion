import torch
import torch.nn as nn

class QuantumDiffusionMLP(nn.Module):
    def __init__(self, vec_len, hidden_dim=1024, num_layers=3):
        super().__init__()
        self.input_dim = (vec_len * 2) + 1

        # Build network
        layers = []
        in_dim = self.input_dim
        for i in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.SiLU()) 
            in_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(hidden_dim, vec_len * 2))

        self.net = nn.Sequential(*layers)

    def forward(self, x, t):
        batch_size = x.shape[0]
        x_flat = x.view(batch_size, -1)

        if t.ndim == 1:
            t = t.unsqueeze(-1)
        elif t.ndim == 0:
            t = t.expand(batch_size, 1)

        combined = torch.cat([x_flat, t.float()], dim=-1)
        out = self.net(combined)
        return out.view(batch_size, 2, -1)
