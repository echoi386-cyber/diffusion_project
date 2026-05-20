# Fourier Equal-SNR Diffusion

## Main scripts

Train:
python scripts/train.py --dataset cifar10 --schedule ddpm ...

Power-law:
python scripts/train.py --dataset cifar10 --schedule power_law --lam 1.0 ...

Export samples:
python scripts/export_samples.py --ckpt ... --outdir ... --num 50000 ...

Create CIFAR test PNGs:
python scripts/create_cifar10_test_png.py

FID:
python src/fourier_diffusion/utils/fid.py ...

Transformed metrics:
python scripts/eval_transformed_metrics.py --real_dir ... --gen_dir ...

## Schedules

ddpm:
Sigma_i = 1

equal_snr:
Sigma_i proportional to C_i

power_law:
Sigma_i proportional to C_i^lambda
