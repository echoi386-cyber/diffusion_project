import torch
from torch.utils.data import DataLoader
from ..models.unet import SimpleUNet
from ..diffusion.forward_process import FourierDiffusionConfig, FourierForwardProcess
from ..diffusion.covariance import estimate_C_diag_rfft2
from ..diffusion.losses import loss_x0_fourier_weighted
from ..diffusion.sampling import ddim_sample

def train_image_dataset(
    name: str,
    loader: DataLoader,
    in_ch: int,
    schedule: Literal["ddpm","equal_snr","flipped_snr"],
    mask_mode: Optional[Literal["none","bernoulli","block","halfplane"]] = "none",
    iters: int = 20000,
    n_C_batches: int = 200,
):
    model = SimpleUNet(in_ch=in_ch, base=64, t_dim=128).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    # Estimate C_diag for equal_snr / flipped_snr
    C_diag = None
    if schedule != "ddpm":
        C_diag = estimate_C_diag_rfft2(loader, device=device, n_batches=n_C_batches)

    fwd = FourierForwardProcess(
        FourierDiffusionConfig(T=cfg.T, schedule=schedule, calibrate_alpha_bar=True),
        C_diag=C_diag,
        device=device,
    )

    loss_log=[]
    loader_iter = iter(loader)

    for step in range(1, iters+1):
        try:
            x0, _ = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            x0, _ = next(loader_iter)
        x0 = x0.to(device, non_blocking=True).float()

        if mask_mode == "bernoulli":
            x0 = mask_pixels_bernoulli(x0, p_zero=0.5)
        elif mask_mode == "block":
            x0 = mask_random_block(x0, frac=0.4)
        elif mask_mode == "halfplane":
            x0 = mask_random_halfplane(x0)

        t = fwd.sample_t(x0.shape[0])
        loss, _ = loss_x0_fourier_weighted(model, fwd, x0, t)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % cfg.log_every == 0:
            loss_log.append(loss.item())
            print(f"[{name} {schedule} mask={mask_mode}] step {step}/{iters} loss {loss.item():.4f}")

        if step % cfg.sample_every == 0:
            model.eval()
            with torch.no_grad():
                samples = ddim_sample(model, fwd, (64, in_ch, x0.shape[-2], x0.shape[-1]), steps=cfg.steps_ddim)
            model.train()
            # spectra compare: real batch vs samples
            real_spec = radial_power_spectrum(x0[:64])
            gen_spec = radial_power_spectrum(samples.clamp(-1,1))
            plot_radial_spectra(
                {"real": real_spec, f"gen_{schedule}": gen_spec},
                f"{name} {schedule} mask={mask_mode} spectrum at step {step}",
                os.path.join(cfg.outdir, f"{name}_{schedule}_{mask_mode}_spec_step{step}.png"),
            )

    return model, fwd, loss_log

def train_toy(schedule: Literal["ddpm","equal_snr","flipped_snr"], iters: int):
    model = ToyMLP(dim=2).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    fwd = ToyForwardProcess(schedule=schedule, T=cfg.T)
    loader_iter = iter(toy_loader)
    for step in range(1, iters+1):
        try:
            (x0_cpu,) = next(loader_iter)
        except StopIteration:
            loader_iter = iter(toy_loader)
            (x0_cpu,) = next(loader_iter)
        x0 = x0_cpu.to(device).float()
        t = fwd.sample_t(x0.shape[0])
        xt, y0, yt = fwd.q_sample(x0, t)
        x0_hat = model(xt, t)
        y0_hat = x0_hat @ fwd.U
        diff = (y0 - y0_hat) / torch.sqrt(fwd.C).view(1,2)
        loss = (diff**2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % cfg.log_every == 0:
            print(f"[toy {schedule}] step {step}/{iters} loss {loss.item():.4f}")
    return model, fwd