import os
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from fourier_diffusion.utils.plots import plot_curve_multi, plot_scatter_2d
from fourier_diffusion.toy.gaussian_utils import (
    sample_diag_gaussian,
    w2_diag,
    kl_diag,
    sliced_wasserstein_1,
)
from fourier_diffusion.toy.gaussian_model import ToyMLP, EMA
from fourier_diffusion.toy.gaussian_forward import GaussianToyForward


@torch.no_grad()
def ddim_sample_toy(model: ToyMLP, fwd: GaussianToyForward, n: int, steps: int) -> torch.Tensor:
    device = fwd.device
    d = fwd.C.numel()

    y = torch.randn(n, d, device=device) * torch.sqrt(fwd.Sigma).view(1, -1)

    ts = torch.linspace(fwd.T, 1, steps, device=device).round().long().unique(sorted=True).flip(0)
    ts = ts.tolist()

    for idx, t_cur in enumerate(ts):
        t = torch.full((n,), t_cur, device=device, dtype=torch.long)
        ab_t = fwd.alpha_bar[t_cur - 1]

        x_t = y @ fwd.U.t()
        x0_hat = model(x_t, t)
        y0_hat = x0_hat @ fwd.U

        eps_hat = (y - torch.sqrt(ab_t) * y0_hat) / torch.sqrt((1.0 - ab_t).clamp_min(1e-12))

        if idx == len(ts) - 1:
            y = y0_hat
        else:
            t_prev = ts[idx + 1]
            ab_prev = fwd.alpha_bar[t_prev - 1]
            y = torch.sqrt(ab_prev) * y0_hat + torch.sqrt((1.0 - ab_prev).clamp_min(1e-12)) * eps_hat

    x = y @ fwd.U.t()
    return x


def train_one(
    schedule: str,
    loader: DataLoader,
    X_all: torch.Tensor,
    U: torch.Tensor,
    C: torch.Tensor,
    device: torch.device,
    T: int,
    iters: int,
    lr: float,
    ema_decay: float,
    c_floor_rel: float,
):
    d = X_all.shape[1]
    model = ToyMLP(dim=d, t_dim=64, width=256, depth=4).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    ema = EMA(model, decay=ema_decay)

    fwd = GaussianToyForward.build(schedule=schedule, U=U, C=C, T=T, device=device)
    C_floor = torch.clamp(C, min=float(c_floor_rel * C.mean().item()))

    model.train()
    it = iter(loader)

    for step in range(1, iters + 1):
        try:
            (x0_cpu,) = next(it)
        except StopIteration:
            it = iter(loader)
            (x0_cpu,) = next(it)

        x0 = x0_cpu.to(device).float()
        t = fwd.sample_t(x0.shape[0])
        xt, y0, _ = fwd.q_sample(x0, t)
        x0_hat = model(xt, t)

        if schedule == "ddpm":
            loss = F.mse_loss(x0_hat, x0)
        else:
            y0_hat = x0_hat @ U
            diff = (y0 - y0_hat) / torch.sqrt(C_floor).view(1, -1)
            loss = (diff ** 2).mean()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        ema.update(model)

        if step % 1000 == 0 or step == 1 or step == iters:
            print(f"[{schedule}] step {step}/{iters} loss {loss.item():.6f}")

    ema_model = ToyMLP(dim=d, t_dim=64, width=256, depth=4).to(device)
    ema.copy_to(ema_model)
    ema_model.eval()
    model.eval()
    return model, ema_model, fwd


@torch.no_grad()
def evaluate_model(
    schedule: str,
    which: str,
    model,
    fwd: GaussianToyForward,
    U: torch.Tensor,
    C: torch.Tensor,
    eval_n: int,
    sw_proj: int,
    outdir: str,
):
    device = C.device

    real_y = sample_diag_gaussian(eval_n, C)
    real_x = real_y @ U.t()

    gen_x = ddim_sample_toy(model, fwd, n=eval_n, steps=min(200, fwd.T))
    gen_y = gen_x @ U

    mean_real = real_x.mean(dim=0)
    mean_gen = gen_x.mean(dim=0)
    std_real = real_x.std(dim=0, unbiased=True)
    std_gen = gen_x.std(dim=0, unbiased=True)

    mean_l2 = float(torch.norm(mean_gen - mean_real).detach().cpu())
    std_l2 = float(torch.norm(std_gen - std_real).detach().cpu())
    sw1 = sliced_wasserstein_1(real_x, gen_x, n_proj=sw_proj)

    C_real_hat = real_y.var(dim=0, unbiased=True).clamp_min(1e-12)
    C_gen_hat = gen_y.var(dim=0, unbiased=True).clamp_min(1e-12)

    w2_spec = w2_diag(C, C_gen_hat)
    kl_spec = kl_diag(C, C_gen_hat)

    idx = torch.arange(1, C.numel() + 1, device=device).detach().cpu().numpy()
    plot_curve_multi(
        idx,
        [
            C.detach().cpu().numpy(),
            C_real_hat.detach().cpu().numpy(),
            C_gen_hat.detach().cpu().numpy(),
        ],
        ["true C", "real empirical", f"gen {schedule} {which}"],
        title=f"Spectrum comparison: {schedule} / {which}",
        xlabel="frequency index i",
        ylabel="variance",
        outpath=os.path.join(outdir, f"spectrum_{schedule}_{which}.png"),
        logy=True,
    )

    plot_scatter_2d(
        real_x[:, :2],
        gen_x[:, :2],
        os.path.join(outdir, f"scatter_x12_{schedule}_{which}.png"),
        title=f"x-space first 2 dims: {schedule} / {which}",
    )

    plot_scatter_2d(
        real_y[:, :2],
        gen_y[:, :2],
        os.path.join(outdir, f"scatter_y12_{schedule}_{which}.png"),
        title=f"frequency-space first 2 dims: {schedule} / {which}",
    )

    return {
        "schedule": schedule,
        "which": which,
        "mean_l2": mean_l2,
        "std_l2": std_l2,
        "sliced_w1": sw1,
        "w2_diag_spectrum": w2_spec,
        "kl_diag_spectrum": kl_spec,
    }


def train_and_evaluate_all(
    loader: DataLoader,
    X_train: torch.Tensor,
    U: torch.Tensor,
    C: torch.Tensor,
    device: torch.device,
    T: int,
    iters: int,
    lr: float,
    ema_decay: float,
    c_floor_rel: float,
    eval_n: int,
    sw_proj: int,
    outdir: str,
):
    metrics_path = os.path.join(outdir, "metrics.txt")
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write("schedule\twhich\ttrain_sec\tmean_l2\tstd_l2\tsliced_w1\tw2_diag_spectrum\tkl_diag_spectrum\n")

    for schedule in ["ddpm", "equal_snr"]:
        t0 = time.time()
        raw_model, ema_model, fwd = train_one(
            schedule=schedule,
            loader=loader,
            X_all=X_train,
            U=U,
            C=C,
            device=device,
            T=T,
            iters=iters,
            lr=lr,
            ema_decay=ema_decay,
            c_floor_rel=c_floor_rel,
        )
        train_sec = time.time() - t0

        for which, model in [("raw", raw_model), ("ema", ema_model)]:
            metrics = evaluate_model(
                schedule=schedule,
                which=which,
                model=model,
                fwd=fwd,
                U=U,
                C=C,
                eval_n=eval_n,
                sw_proj=sw_proj,
                outdir=outdir,
            )

            print(
                f"[gaussian toy {schedule} {which}] "
                f"train_time={train_sec:.1f}s  "
                f"mean_l2={metrics['mean_l2']:.6f}  "
                f"std_l2={metrics['std_l2']:.6f}  "
                f"SW1={metrics['sliced_w1']:.6f}  "
                f"W2diag={metrics['w2_diag_spectrum']:.6f}  "
                f"KLdiag={metrics['kl_diag_spectrum']:.6f}"
            )

            with open(metrics_path, "a", encoding="utf-8") as f:
                f.write(
                    f"{schedule}\t{which}\t{train_sec:.6f}\t"
                    f"{metrics['mean_l2']:.6f}\t{metrics['std_l2']:.6f}\t"
                    f"{metrics['sliced_w1']:.6f}\t"
                    f"{metrics['w2_diag_spectrum']:.6f}\t"
                    f"{metrics['kl_diag_spectrum']:.6f}\n"
                )