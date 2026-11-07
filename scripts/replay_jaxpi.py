"""Train a Table 2 row from inputs exported from jaxpi: its initial network and, optionally, its batches.

This reruns the paper's own experiment in this trainer. The paper's runs all start from jaxpi's seed 42 and
train on one batch sequence (jaxpi's sampler uses a fixed key), so with both exported (scripts/jaxpi/),
the only thing left to differ is the trainer.

    python scripts/replay_jaxpi.py --row default --init init_default.npz --batches jaxpi_batches.npy
    python scripts/replay_jaxpi.py --row no_fourier --init init_no_fourier.npz --batches jaxpi_batches.npy

--init must come from the matching jaxpi config (export_init.py --config, see JAXPI_CONFIG below).
Without --batches the run draws its own batches. --ic-grid jaxpi (the default) fits the initial condition
on jaxpi's x-grid, linspace(-1, 1, 512), instead of this code's 512 periodic points on [-1, 1).

The run writes to runs/allen_cahn/paper_replay/<row>/ (or --out-dir) like scripts/train.py, plus
replay.json naming its inputs. Started again, it resumes from its checkpoint; the inputs must not change.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from pinnguide.config import load_config, resolve_path
from pinnguide.jaxpi_interop import JAXPI_IC_X, BatchFile, load_jaxpi_params
from pinnguide.trainer import Trainer

ROWS = ["default", "no_causal", "no_rwf", "no_grad_norm", "no_fourier", "plain"]
# The jaxpi config (examples/allen_cahn/configs/<name>.py) whose network each row starts from.
# no_grad_norm builds the same network as default.
JAXPI_CONFIG = {"default": "default", "no_causal": "no_causal", "no_rwf": "no_rwf", "no_grad_norm": "default",
                "no_fourier": "no_fourier_feature", "plain": "plain"}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--row", choices=ROWS, default="default")
    p.add_argument("--init", required=True, help="initial network from scripts/jaxpi/export_init.py")
    p.add_argument("--batches", help="batch sequence from scripts/jaxpi/export_batches.py")
    p.add_argument("--ic-grid", choices=["jaxpi", "reference"], default="jaxpi")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override config values")
    p.add_argument("--out-dir", help="default: runs/allen_cahn/paper_replay/<row>")
    p.add_argument("--max-steps", type=int, help="stop after this many steps in this session")
    p.add_argument("--check-only", action="store_true", help="load the network, check it against jaxpi's, exit")
    a = p.parse_args()

    # The paper's seed. With the network and batches loaded it no longer matters, but it keeps the saved
    # config honest about which run this repeats.
    cfg = load_config(f"configs/allen_cahn/ablation/{a.row}.yaml", ["training.seed=42", *a.set])
    cfg.logging.out_dir = a.out_dir or f"runs/allen_cahn/paper_replay/{a.row}"
    if cfg.training.ic_sampling != "grid":
        raise SystemExit("replaying jaxpi needs training.ic_sampling=grid")

    inputs = {"init": str(Path(a.init).resolve()), "batches": str(Path(a.batches).resolve()) if a.batches else None,
              "ic_grid": a.ic_grid}
    trainer = Trainer(cfg)
    record = cfg.out_dir / "replay.json"
    resuming = trainer.checkpoint_path().exists()
    if resuming and record.exists():
        # A resumed run takes its weights from the checkpoint, so only the batches and the IC grid must match.
        saved = json.loads(record.read_text(encoding="utf-8"))
        changed = [k for k in ("batches", "ic_grid") if saved.get(k) != inputs[k]]
        if changed:
            raise SystemExit(f"{record} was started with different inputs ({', '.join(changed)}): {saved}")

    if a.ic_grid == "jaxpi":
        trainer.x_ic = torch.as_tensor(JAXPI_IC_X, device=trainer.device)
    if a.batches:
        batches = BatchFile(resolve_path(a.batches), cfg.causal.num_chunks, cfg.causal.enabled, trainer.device)
        if batches.batch_size != cfg.training.batch_size:
            raise SystemExit(f"the batch file holds batches of {batches.batch_size}, the config "
                             f"{cfg.training.batch_size}")
        if len(batches) < cfg.training.steps:
            print(f"warning: the batch file holds {len(batches)} steps, fewer than the {cfg.training.steps} "
                  "configured; use --max-steps or --set training.steps", flush=True)
        trainer.sample_batch = lambda: (*batches(trainer.step), trainer.x_ic)
        print(f"batches: {a.batches}, {len(batches)} steps of {batches.batch_size}", flush=True)
    print(f"IC grid: {a.ic_grid}, {trainer.x_ic.numel()} points from {trainer.x_ic[0].item():.4f} "
          f"to {trainer.x_ic[-1].item():.4f}", flush=True)

    if not resuming:
        z = np.load(resolve_path(a.init))
        load_jaxpi_params(trainer.model, z)
        if "check_u" in z.files:
            with torch.no_grad():
                u = trainer.model(torch.as_tensor(z["check_t"], device=trainer.device),
                                  torch.as_tensor(z["check_x"], device=trainer.device)).cpu().numpy()
            print(f"initial network: max |u - u_jaxpi| = {np.abs(u - z['check_u']).max():.2e} at "
                  f"{len(u)} points (|u| up to {np.abs(z['check_u']).max():.2f})", flush=True)
    if a.check_only:
        return
    if not resuming:
        cfg.out_dir.mkdir(parents=True, exist_ok=True)
        record.write_text(json.dumps(inputs, indent=2), encoding="utf-8")
    trainer.train(resume=resuming, max_steps=a.max_steps)


if __name__ == "__main__":
    main()
