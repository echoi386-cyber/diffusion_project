#!/usr/bin/env python3
import os
import subprocess

def main():
    outroot = "outputs"
    os.makedirs(outroot, exist_ok=True)

    for sched in ["ddpm", "equal_snr"]:
        subprocess.check_call([
            "python", "scripts/train.py",
            "--dataset", "mnist",
            "--schedule", sched,
            "--iters", "50000",
            "--batch_size", "128",
            "--sample_every", "5000",
            "--steps_ddim", "200",
            "--num_fid_samples", "10000",
            "--outdir", outroot,
        ])

if __name__ == "__main__":
    main()