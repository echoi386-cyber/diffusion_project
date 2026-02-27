#!/usr/bin/env python3
import os
import subprocess

def main():
    outroot = "outputs"
    os.makedirs(outroot, exist_ok=True)

    for sched in ["ddpm", "equal_snr", "flipped_snr"]:
        subprocess.check_call([
            "python", "scripts/train.py",
            "--dataset", "cifar10",
            "--schedule", sched,
            "--iters", "8000",
            "--batch_size", "128",
            "--outdir", outroot
        ])

if __name__ == "__main__":
    main()