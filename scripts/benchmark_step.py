"""Time training steps on the current device and project the run time of a full run.

    python scripts/benchmark_step.py --config configs/allen_cahn/ablation/default.yaml

Times the two derivative implementations (training.derivatives = autograd | func) and the
compiled step (training.compile = true). All three give the same values up to float32
round-off (tests/test_derivatives.py, tests/test_trainer.py).
"""
import argparse
import tempfile
import time

import torch

from pinnguide.config import load_config
from pinnguide.trainer import Trainer

VARIANTS = {
    "autograd": ["training.derivatives=autograd", "training.compile=false"],
    "func": ["training.derivatives=func", "training.compile=false"],
    "compile": ["training.compile=true"],
}


def time_steps(cfg, n_warmup: int, n_steps: int) -> float:
    trainer = Trainer(cfg)
    for _ in range(n_warmup):  # the first compiled step also compiles
        trainer.train_step()
    if trainer.device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_steps):
        trainer.train_step()
    if trainer.device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - start) / n_steps


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/allen_cahn/ablation/default.yaml")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--only", nargs="*", choices=list(VARIANTS), help="variants to time (default: all)")
    a = p.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        for name in a.only or VARIANTS:
            cfg = load_config(a.config, a.set + VARIANTS[name])
            cfg.logging.out_dir = tmp
            try:
                sec = time_steps(cfg, a.warmup, a.steps)
            except Exception as e:  # e.g. torch.compile without Triton
                print(f"{name:>8s}: unavailable ({type(e).__name__}: {str(e).splitlines()[0]})", flush=True)
                continue
            minutes = sec * cfg.training.steps / 60
            print(f"{name:>8s}: {sec * 1e3:6.2f} ms/step -> {minutes:5.1f} min per {cfg.training.steps}-step run "
                  "(training steps only)", flush=True)
    if torch.cuda.is_available():
        print("device:", torch.cuda.get_device_name())


if __name__ == "__main__":
    main()
