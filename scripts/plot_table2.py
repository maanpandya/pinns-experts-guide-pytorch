"""Figure for docs/results.md: Table 2 of the Expert's Guide as distributions of final rel. L2.

Left panel: runs from the paper's own inputs (the seed-42 starting network and batch sequence of the paper's
runs), the paper's logged run against the authors' code (jaxpi) and ours. Right panel: runs from other seeds,
ours and jaxpi's, with the paper's single run for reference. The left panel's values and jaxpi's were measured
with the authors' code and its exported inputs (docs/results.md); the paper's are Table 2's.

Our own runs are every finished runs/allen_cahn/*/ablation_<row>/ whose saved config matches the row's
YAML in configs/allen_cahn/ablation/ (apart from the seed), one per seed. Make more with, for example,
    python scripts/run_ablation.py --set training.seed=1 --out-root runs/allen_cahn/seed1

usage: python scripts/plot_table2.py [--runs runs/allen_cahn] [--out docs/table2.png]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from pinnguide.config import load_config
from pinnguide.trainer import config_mismatch
from plot_style import INK, INK_2, SERIES, apply_style, plt

ROWS = ["default", "no_causal", "no_rwf", "no_grad_norm", "no_fourier", "plain"]
PAPER = {"default": 5.84e-4, "no_causal": 1.59e-3, "no_rwf": 6.62e-3, "no_grad_norm": 7.51e-3,
         "no_fourier": 4.35e-1, "plain": 5.17e-1}
# The authors' code (jaxpi) from seed 42 on its own batches, the paper's runs' inputs, at full float32 precision.
JAXPI_42 = {"default": 5.52e-4, "no_causal": 1.29e-3, "no_rwf": 3.83e-3, "no_grad_norm": 7.72e-3,
            "no_fourier": 4.74e-2, "plain": 5.03e-1}
# Ours from the same inputs (the paper's network and batches, exported from the authors' code).
REPLAY = {"default": 5.56e-4, "no_causal": 5.26e-3, "no_rwf": 6.72e-3, "no_grad_norm": 6.66e-3,
          "no_fourier": 7.37e-2, "plain": 4.85e-1}
# jaxpi from seeds 0-5: default on its sampler's fixed key and on one other key per seed; no_fourier on the fixed key.
JAXPI_SEEDS = {
    "default": [1.51e-3, 1.39e-3, 1.73e-3, 2.89e-3, 2.03e-3, 4.05e-3, 1.80e-3, 2.16e-3, 1.60e-3, 2.31e-3, 2.90e-3, 2.39e-3],
    "no_fourier": [5.08e-1, 4.15e-1, 4.32e-2, 4.81e-1, 3.66e-2, 3.27e-2],
}
CONFIGS = Path(__file__).resolve().parents[1] / "configs/allen_cahn/ablation"


def own_seeds(runs, row):
    """Final rel. L2 of the finished runs of this row made with its current config, one per seed."""
    cfg = load_config(CONFIGS / f"{row}.yaml")
    found = {}
    for summary in sorted(runs.glob(f"*/ablation_{row}/summary.json")):
        s = json.loads(summary.read_text(encoding="utf-8"))
        saved_path = summary.parent / "config.yaml"
        if not (s.get("finished") and s.get("step") == cfg.training.steps and saved_path.exists()):
            continue
        saved = yaml.safe_load(saved_path.read_text(encoding="utf-8"))
        cfg.training.seed = saved["training"]["seed"]
        if not config_mismatch(saved, cfg):
            found.setdefault(cfg.training.seed, s["rel_l2"])
    return list(found.values())


def strip(ax, x, values, color, label, width=0.2):
    """Dots in three staggered columns by rank, so that close values stay visible without reading as a trend."""
    v = np.sort(values)
    dx = (np.arange(len(v)) % 3 - 1) * width / 2 if len(v) > 1 else np.zeros(1)
    ax.scatter(x + dx, v, s=34, color=color, edgecolor="white", linewidth=0.8, zorder=3, label=label)


def counts(ax, x, ns):
    """Run counts of a row's strips, left to right, on one line above the x axis."""
    ax.text(x, 0.03, "n = " + ", ".join(map(str, ns)), transform=ax.get_xaxis_transform(), ha="center",
            fontsize=7, color=INK_2)


def paper_tick(ax, x, y, label=None, half=0.42):
    ax.plot([x - half, x + half], [y, y], color=INK, linewidth=1.6, solid_capstyle="round", zorder=2, label=label)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="runs/allen_cahn")
    p.add_argument("--out", default="docs/table2.png")
    a = p.parse_args()
    runs = Path(a.runs)
    apply_style()
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    xs = np.arange(len(ROWS))
    for i, row in enumerate(ROWS):
        paper_tick(left, i, PAPER[row], "paper (one run, seed 42)" if i == 0 else None)
        strip(left, i - 0.2, [JAXPI_42[row]], SERIES[1], "jaxpi" if i == 0 else None, 0)
        strip(left, i + 0.2, [REPLAY[row]], SERIES[0], "ours" if i == 0 else None, 0)
        paper_tick(right, i, PAPER[row], "paper (one run, seed 42)" if i == 0 else None)
        if row in JAXPI_SEEDS:
            strip(right, i - 0.2, JAXPI_SEEDS[row], SERIES[1], "jaxpi, seeds 0–5" if i == 0 else None)
        ours = own_seeds(runs, row)
        if ours:
            strip(right, i + 0.2, ours, SERIES[0], "ours, own seeds" if i == 0 else None)
        ns = [len(JAXPI_SEEDS[row])] if row in JAXPI_SEEDS else []
        ns += [len(ours)] if ours else []
        if ns:
            counts(right, i if len(ns) > 1 else i + 0.2, ns)
    left.set_title("From the paper's inputs (its network and batches)", loc="left")
    right.set_title("From other seeds", loc="left")
    left.set_ylabel("rel. L2 error after 200k steps")
    for ax in (left, right):
        ax.set_yscale("log")
        ax.set_ylim(2e-4, 1)
        ax.set_xticks(xs, ROWS, rotation=20, ha="right")
        ax.set_xlim(-0.6, len(ROWS) - 0.4)
        ax.grid(axis="x", visible=False)
        ax.legend(loc="upper left", fontsize=8, handlelength=1.2)
    fig.tight_layout()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
