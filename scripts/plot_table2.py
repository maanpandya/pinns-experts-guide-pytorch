"""Figure for docs/results.md: Table 2 of the Expert's Guide as distributions of final rel. L2.

Left panel: runs from the paper's own inputs (jaxpi's seed-42 network and its batch sequence), the paper's
logged run against jaxpi's and ours. Right panel: runs from other seeds, ours and jaxpi's, with the paper's
single run for reference. Our values are read from the run folders; the paper's are Table 2's, and jaxpi's
were measured by running jaxpi outside this repository (handoff.md, "jaxpi itself").

usage: python scripts/plot_table2.py [--runs runs/allen_cahn] [--out docs/table2.png]
"""
import argparse
import json
from pathlib import Path

import numpy as np

from plot_style import INK, INK_2, SERIES, apply_style, plt

ROWS = ["default", "no_causal", "no_rwf", "no_grad_norm", "no_fourier", "plain"]
PAPER = {"default": 5.84e-4, "no_causal": 1.59e-3, "no_rwf": 6.62e-3, "no_grad_norm": 7.51e-3,
         "no_fourier": 4.35e-1, "plain": 5.17e-1}
# jaxpi from seed 42 on its own (the paper's) batches, at full float32 precision.
JAXPI_42 = {"default": 5.52e-4, "no_causal": 1.29e-3, "no_rwf": 3.83e-3, "no_grad_norm": 7.72e-3,
            "no_fourier": 4.74e-2, "plain": 5.03e-1}
# jaxpi from seeds 0-5: default on sampler key 1234 and on key 100 + seed; no_fourier on key 1234.
JAXPI_SEEDS = {
    "default": [1.51e-3, 1.39e-3, 1.73e-3, 2.89e-3, 2.03e-3, 4.05e-3, 1.80e-3, 2.16e-3, 1.60e-3, 2.31e-3, 2.90e-3, 2.39e-3],
    "no_fourier": [5.08e-1, 4.15e-1, 4.32e-2, 4.81e-1, 3.66e-2, 3.27e-2],
}
# Rows with RWF use jaxpi's parametrization (scale_param direct, D21); no_rwf and plain have no RWF.
RWF_ROWS = {"default", "no_causal", "no_grad_norm", "no_fourier"}


def final(path):
    with open(path) as f:
        return json.load(f)["rel_l2"]


def own_seeds(runs, row):
    if row in RWF_ROWS:
        paths = sorted(runs.glob(f"rwf_direct_seed*/ablation_{row}/summary.json"))
    else:
        paths = [runs / d / f"ablation_{row}" / "summary.json" for d in ("ablation", "ablation_seed1", "ablation_seed2")]
    return [final(p) for p in paths if p.exists() and json.load(open(p)).get("finished")]


def strip(ax, x, values, color, label, width=0.2):
    """Dots in three staggered columns by rank, so that close values stay visible without reading as a trend."""
    v = np.sort(values)
    dx = (np.arange(len(v)) % 3 - 1) * width / 2 if len(v) > 1 else np.zeros(1)
    ax.scatter(x + dx, v, s=34, color=color, edgecolor="white", linewidth=0.8, zorder=3, label=label)
    if len(v) > 1:  # run counts on one line above the x axis, under their strip
        ax.text(x, 0.03, f"n={len(v)}", transform=ax.get_xaxis_transform(), ha="center", fontsize=7, color=INK_2)


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
        replay = runs / "paper_replay" / row / "summary.json"
        if replay.exists():
            strip(left, i + 0.2, [final(replay)], SERIES[0], "ours" if i == 0 else None, 0)
        paper_tick(right, i, PAPER[row], "paper (one run, seed 42)" if i == 0 else None)
        if row in JAXPI_SEEDS:
            strip(right, i - 0.2, JAXPI_SEEDS[row], SERIES[1], "jaxpi, seeds 0–5" if i == 0 else None)
        ours = own_seeds(runs, row)
        if ours:
            strip(right, i + 0.2, ours, SERIES[0], "ours, own seeds" if i == 0 else None)
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
