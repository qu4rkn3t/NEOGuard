import torch
import torch.nn as nn
from typing import Tuple
from physics import two_body_acc, drag_acc, srp_acc


class TinyEncoder(nn.Module):
    def __init__(self, in_dim=6, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualHead(nn.Module):
    def __init__(self, hidden=128, out_dim=6):
        super().__init__()
        self.head = nn.Linear(hidden, out_dim)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.head(h)


class OrbitPINN(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.enc = TinyEncoder(6, hidden)
        self.head = ResidualHead(hidden, 6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        x2 = x.reshape(B * T, D)
        h = self.enc(x2)
        out = self.head(h).reshape(B, T, D)
        return out


def physics_residual(
    states: torch.Tensor, dt: float = 60.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    r = states[..., :3]
    v = states[..., 3:]
    rp = (r[..., 1:, :] - r[..., :-1, :]) / dt
    vp = (v[..., 1:, :] - v[..., :-1, :]) / dt
    r_mid = (r[..., 1:, :] + r[..., :-1, :]) * 0.5
    v_mid = (v[..., 1:, :] + v[..., :-1, :]) * 0.5
    a = two_body_acc(r_mid) + drag_acc(r_mid, v_mid) + srp_acc(r_mid)
    r_res = rp - v_mid
    v_res = vp - a
    return r_res, v_res
