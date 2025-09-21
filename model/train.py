import torch
import argparse
from tqdm import tqdm
from pathlib import Path
from pinn import OrbitPINN, physics_residual
from datasets import prepare_training_data


def train(norad: int, minutes: int, epochs: int, lr: float, hidden: int, api_key: str):
    x, y = prepare_training_data(norad, api_key=api_key, minutes=minutes)
    x = x.unsqueeze(0)
    y = y.unsqueeze(0)
    model = OrbitPINN(hidden=hidden)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    mse = torch.nn.MSELoss()
    for ep in tqdm(range(1, epochs + 1)):
        model.train()
        opt.zero_grad()
        delta = model(x)
        states = x + delta
        sup_loss = mse(states, y) * 0.1
        r_res, v_res = physics_residual(states, dt=60.0)
        phys_loss = r_res.pow(2).mean() + v_res.pow(2).mean()
        loss = sup_loss + phys_loss
        loss.backward()
        opt.step()
        if ep % 250 == 0:
            print(f"epoch={ep} loss={loss.item():.6f} phys={phys_loss.item():.6f}")
    ckpt = Path("checkpoints")
    ckpt.mkdir(parents=True, exist_ok=True)
    path = ckpt / "model.pt"
    torch.jit.script(model).save(str(path))
    print(f"Saved model to {path.resolve()}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--norad", type=int, default=25544)
    p.add_argument("--minutes", type=int, default=360)
    p.add_argument("--epochs", type=int, default=2000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=128)
    args = p.parse_args()
    api_key = "kxzkc6cT0B7cVo2K6zhybV9tqFTHfY85ofBShxsz"
    train(args.norad, args.minutes, args.epochs, args.lr, args.hidden, api_key=api_key)
