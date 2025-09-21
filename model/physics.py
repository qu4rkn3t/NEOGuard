import torch

MU_EARTH = 398600.4418
R_EARTH = 6378.1363
CD = 2.2
AOM = 0.01
H_SCALE = 60.0


def atmospheric_density(h_km: torch.Tensor) -> torch.Tensor:
    rho0 = 4e-12
    return rho0 * torch.exp(-h_km / H_SCALE)


def two_body_acc(r: torch.Tensor) -> torch.Tensor:
    norm_r = torch.linalg.norm(r, dim=-1, keepdim=True) + 1e-9
    return -MU_EARTH * r / (norm_r**3)


def drag_acc(r: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    norm_r = torch.linalg.norm(r, dim=-1, keepdim=True)
    h = norm_r - R_EARTH
    rho = atmospheric_density(h)
    v_ms = v * 1000.0
    vmag = torch.linalg.norm(v_ms, dim=-1, keepdim=True) + 1e-9
    a_ms2 = -0.5 * CD * AOM * rho * vmag * v_ms
    return a_ms2 / 1000.0


def srp_acc(r: torch.Tensor) -> torch.Tensor:
    P0 = 4.5e-6
    CrAoverm = 0.01
    a_ms2 = (P0 * CrAoverm) * (r / (torch.linalg.norm(r, dim=-1, keepdim=True) + 1e-9))
    return a_ms2 / 1000.0
