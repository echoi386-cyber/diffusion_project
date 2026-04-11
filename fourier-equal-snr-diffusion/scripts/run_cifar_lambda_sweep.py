#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type=str, required=True)
    p.add_argument("--lams", type=str, required=True)  # e.g. "0,0.5,1.0,1.5"
    p.add_argument("--iters", type=int, default=800000)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--steps_ddim", type=int, default=200)
    p.add_argument("--outroot", type=str, default="outputs")
    args = p.parse_args()

    lams = [x.strip() for x in args.lams.split(",") if x.strip()]
    os.makedirs(args.outroot, exist_ok=True)
    os.makedirs(os.path.join(args.outroot, "logs"), exist_ok=True)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu

    for lam in lams:
        outdir = os.path.join(args.outroot, f"cifar_lambda_{lam}")
        logfile = os.path.join(args.outroot, "logs", f"cifar_lambda_{lam}.log")

        cmd = [
            sys.executable, "-u", "scripts/train.py",
            "--dataset", "cifar10",
            "--schedule", "power_law",
            "--lam", lam,
            "--calibration", "fixed_trace",
            "--iters", str(args.iters),
            "--batch_size", str(args.batch_size),
            "--lr", "2e-4",
            "--T", "1000",
            "--sample_every", "50000",
            "--save_every", "10000",
            "--steps_ddim", str(args.steps_ddim),
            "--outdir", outdir,
        ]

        print("Running:", " ".join(cmd))
        with open(logfile, "w", encoding="utf-8") as f:
            subprocess.check_call(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)


if __name__ == "__main__":
    main()