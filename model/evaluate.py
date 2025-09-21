import argparse, os
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from datasets import prepare_training_data
from pinn import OrbitPINN, physics_residual


def load_checkpoint(path: str) -> torch.nn.Module:
    m = torch.jit.load(path, map_location="cpu")
    m.eval()
    return m


def physics_metrics(states: torch.Tensor):
    r_res, v_res = physics_residual(states, dt=60.0)
    return float(r_res.pow(2).mean().sqrt()), float(v_res.pow(2).mean().sqrt())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--norad", type=int, default=25544)
    ap.add_argument("--minutes", type=int, default=360)
    ap.add_argument("--checkpoint", type=str, default="ml/checkpoints/model.pt")
    args = ap.parse_args()

    api_key = os.environ.get("NASA_API_KEY", "DEMO_KEY")

    x, _ = prepare_training_data(args.norad, api_key=api_key, minutes=args.minutes)
    x = x.unsqueeze(0)

    b_r, b_v = physics_metrics(x)

    if not Path(args.checkpoint).exists():
        print("Checkpoint not found; evaluating baseline only.")
        print(f"Baseline residuals: r'−v={b_r:.4e}, v'−a={b_v:.4e}")
        return

    model = load_checkpoint(args.checkpoint)
    with torch.no_grad():
        delta = model(x)
        states = x + delta

    p_r, p_v = physics_metrics(states)

    print("=== Physics Residuals (lower is better) ===")
    print(f"Baseline: r'−v={b_r:.4e}, v'−a={b_v:.4e}")
    print(f"PINN    : r'−v={p_r:.4e}, v'−a={p_v:.4e}")

    t = np.arange(x.shape[1]) * 60.0
    vb = torch.linalg.norm(x[0, :, 3:], dim=-1).numpy()
    vp = torch.linalg.norm(states[0, :, 3:], dim=-1).numpy()
    plt.figure()
    plt.plot(t / 60.0, vb, label="Baseline SGP4 speed (km/s)")
    plt.plot(t / 60.0, vp, label="PINN-corrected speed (km/s)", linestyle="--")
    plt.xlabel("Time (min)")
    plt.ylabel("Speed (km/s)")
    plt.legend()
    out = Path("ml/eval_speed.png")
    plt.savefig(out, dpi=160, bbox_inches="tight")
    print(f"Saved {out.resolve()}")


if __name__ == "__main__":
    main()
    out = Path("ml/eval_speed.png")
    plt.savefig(out, dpi=160, bbox_inches="tight")
    print(f"Saved {out.resolve()}")


if __name__ == "__main__":
    main()
