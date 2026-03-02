import os
import numpy as np
import torch
import matplotlib.pyplot as plt

@torch.no_grad()
def eval_residual_plots(model, fwd, X_eval, outdir: str, tag: str, t_values=(200, 800)):
    os.makedirs(outdir, exist_ok=True)
    model.eval()
    X_eval = X_eval.to(fwd.device).float()

    for t0 in t_values:
        t = torch.full((X_eval.shape[0],), int(t0), device=fwd.device, dtype=torch.long)
        xt, y0 = fwd.q_sample(X_eval, t)
        x0_hat = model(xt, t)
        y0_hat = x0_hat @ fwd.U

        resid = (y0 - y0_hat)  # (N,D)
        mse_dir = (resid ** 2).mean(dim=0).detach().cpu().numpy()  # (D,)

        # Plot mse_dir vs index
        plt.figure()
        plt.plot(np.arange(len(mse_dir)), mse_dir)
        plt.title(f"MSE per PCA direction vs index (t={t0}) [{tag}]")
        plt.xlabel("direction index i (PCA ~ frequency-like)")
        plt.ylabel("E[(y0_i - y0hat_i)^2]")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"mse_dir_t{t0}_{tag}.png"))
        plt.close()

        # Histogram of log10 mse_dir
        plt.figure()
        plt.hist(np.log10(mse_dir + 1e-12), bins=60)
        plt.title(f"Histogram of log10 MSE per direction (t={t0}) [{tag}]")
        plt.xlabel("log10 MSE_i")
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"hist_log_mse_dir_t{t0}_{tag}.png"))
        plt.close()