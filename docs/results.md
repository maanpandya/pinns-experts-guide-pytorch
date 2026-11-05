# Results: the Expert's Guide Allen–Cahn ablation (Table 2)

Relative L2 error after 200k steps, for each row of Table 2 in the Expert's Guide (arXiv:2308.08468). Each
row turns off one component of the training pipeline. All runs are float32, with TF32 off, on one RTX 3070 Ti
Laptop GPU, at about 1 h per run.

## Summary

- **Given the paper's starting network and batches, our trainer lands on the paper's values.** Four rows end
  within 0.89–1.01× of the paper. Of the other two, `no_fourier` and `no_causal`, each run can end in one of
  two ways, and the paper's single run and ours went different ways. On `no_fourier`, jaxpi's own run from
  the same inputs went our way.
- **From our own seeds, `default` ends 4.5× above the paper by median, because the paper's run was lucky.**
  The paper's values are single runs from seed 42. jaxpi, the authors' code, run on seeds 0–5, ends 3.6×
  above the paper on `default` by median (12 runs, on two batch sequences per seed). Our ten seeds end
  where jaxpi's runs do: their geometric mean is 1.05× jaxpi's (Mann–Whitney p = 0.34). They range from
  3.6e-4, below the paper, to 4.8e-3. From jaxpi's own seed 0–5 networks, our trainer ends where jaxpi's
  does, on jaxpi's batches and on our own (18 and 12 runs, both at a geometric mean of 2.01e-3).
- By median, our rows rank as the paper's do, except that `no_rwf` and `no_grad_norm` swap; they are within
  15% of each other in the paper.

![Final rel. L2 of each Table 2 row, as distributions](table2.png)

Final rel. L2 of each row after 200k steps. Left: from the paper's own network and batch sequence. Right:
from other seeds, jaxpi's (seeds 0–5; for `default` on two batch sequences each) and ours. The black tick
is the paper's single run, and n gives the number of runs in each strip, left to right.
`scripts/plot_table2.py` draws it from the run folders.

## Our runs, from our own seeds

| Row | Fourier features | RWF | Grad norm | Causal | Seeds 0, 1, 2 | Median | Paper | Median / paper |
|---|:-:|:-:|:-:|:-:|---|--:|--:|--:|
| `default` | yes | yes | yes | yes | 2.93e-3, 2.27e-3, 2.27e-3 | 2.60e-3 (10 seeds) | 5.84e-4 | 4.5 |
| `no_causal` | yes | yes | yes | no | 5.61e-3, 4.95e-3, 5.16e-3 | 5.16e-3 | 1.59e-3 | 3.2 |
| `no_rwf` | yes | no | yes | yes | 1.39e-2, 1.20e-2, 7.60e-3 | 1.20e-2 | 6.62e-3 | 1.8 |
| `no_grad_norm` | yes | yes | no | yes | 2.86e-3, 6.41e-3, 6.14e-3 | 6.14e-3 | 7.51e-3 | 0.82 |
| `no_fourier` | no | yes | yes | yes | 5.12e-1, 5.06e-1, 4.24e-1 | 5.06e-1 | 4.35e-1 | 1.2 |
| `plain` | no | no | no | no | 5.25e-1, 4.94e-1, 5.08e-1 | 5.08e-1 | 5.17e-1 | 0.98 |

`default` also has seeds 3–9: 3.67e-3, 4.81e-3, 1.07e-3, 3.64e-4, 1.72e-3, 4.14e-3 and 4.09e-3. Its
median is over all ten, and seed 6 ends below the paper. `scripts/run_ablation.py` produces these runs with
the configs in `configs/allen_cahn/ablation/`.

## The paper's runs, replayed in our trainer

The authors' runs for Table 2 are public on Weights & Biases (project `jaxpi/allen_cahn`), logged every 100
steps. They all start from jaxpi's seed 42. jaxpi's residual sampler uses a fixed key whatever the seed, so
they also all train on one batch sequence. For comparison, we ran jaxpi locally and exported each row's initial
weights and that batch sequence, as plain numbers. We then trained our pipeline on them for all 200k steps.
These runs repeat the paper's experiment in our code, and isolate the trainer from the luck of the seed.

| Row | Paper | jaxpi, same network and batches | Ours, same network and batches | Ours / paper |
|---|--:|--:|--:|--:|
| `default` | 5.84e-4 | 5.52e-4 | 5.56e-4 | 0.95 |
| `no_causal` | 1.59e-3 | 1.29e-3 | 5.26e-3 | 3.31 |
| `no_rwf` | 6.62e-3 | 3.83e-3 | 6.72e-3 | 1.01 |
| `no_grad_norm` | 7.51e-3 | 7.72e-3 | 6.66e-3 | 0.89 |
| `no_fourier` | 4.35e-1 | 4.74e-2 | 7.37e-2 | 0.17 |
| `plain` | 5.17e-1 | 5.03e-1 | 4.85e-1 | 0.94 |

- **`default`, `no_rwf`, `no_grad_norm` and `plain`** follow the paper's runs throughout training. For
  example, `default` reaches rel. L2 9.4e-4 at 100k steps, the same value as jaxpi's run on the same inputs.
- **Runs on the same inputs still drift apart.** jaxpi's own `no_rwf` run ends 1.7× below the paper's, from
  the same network and batches.
- **`no_causal` ends high because of one setback.** Our run follows jaxpi's and the paper's runs until
  19.7k steps. Then a single burst in the losses throws it back to the early plateau (rel. L2 0.50), and it
  stays behind to the end.
  - Such setbacks happen in jaxpi too. From the same network, on ten different batch sequences, jaxpi fell
    back in 1 of 10 runs of 30k steps, and our trainer in 3 of 11 (one-sided Fisher p = 0.33).
  - The bursts behind them are about as frequent in both trainers. Above 30× the recent median, ours has
    slightly more (22 against 19 per run); on `default` it had fewer. The runs with the most bursts did not
    fall back.
- **`no_fourier` has two outcomes.** A run either stays at a wrong solution (rel. L2 0.4–0.5), as the
  paper's run does, or leaves it for the right one (0.03–0.08).
  - jaxpi's run from the paper's exact network and batches leaves it, as our replay does. It matches the
    paper's run at 1k steps, escapes from 39k, and ends at 4.74e-2. So the paper's run stayed by chance.
  - jaxpi escapes in 4 of 7 runs of this row (seeds 42 and 0–5), and our trainer in 8 of 14 (Fisher
    p = 1.0). Six of ours start from jaxpi's seed 0–5 networks on jaxpi's batches.
  - Once escaped, both end at the same level: ours at 2.9–8.1e-2, jaxpi's at 3.3–4.7e-2 (Mann–Whitney
    p = 0.28). The six from jaxpi's networks and batches end at 2.9–5.8e-2 (p = 0.90).
  - Over the first 100 updates from the same inputs, the two trainers' IC losses agree to 1e-5. But this
    row amplifies round-off quickly: two runs of our own trainer, on CPU and GPU, are 28% apart within 300
    steps. So from the same inputs, each run is an independent draw. Ours escaped from jaxpi's seed 0, 1, 3
    networks, where jaxpi's runs did not, and jaxpi's escaped from seed 5, where ours stalled at 0.24.

## Why our own seeds land above the paper

- **The paper's seed was favorable.** jaxpi's own `default` config, run unmodified, reproduces the paper from
  seed 42 (5.52e-4). Seeds 0–5 give 1.39–4.05e-3 instead on the paper's batch sequence, and 1.60–2.90e-3 on
  others. The median of those 12 runs is 2.1e-3, 3.6× the paper.
- **Given the same inputs, our trainer and jaxpi's agree.** On the same network and batch, the two
  trainers' losses agree to 8 digits and their loss weights to 6, and every parameter moves the same way in
  the first update. Over a whole run, given the same starting network and the same batch sequence:

  | jaxpi's network | jaxpi | Ours | Ours / jaxpi |
  |---|--:|--:|--:|
  | seed 42 | 5.52e-4 | 5.56e-4 | 1.01 |
  | seed 0 | 1.51e-3 | 1.53e-3 | 1.01 |
  | seed 1 | 1.39e-3 | 1.09e-3 | 0.78 |
  | seed 2 | 1.73e-3 | 1.77e-3 | 1.02 |
  | seed 3 | 2.89e-3 | 3.12e-3 | 1.08 |
  | seed 4 | 2.03e-3 | 2.29e-3 | 1.13 |
  | seed 5 | 4.05e-3 | 1.61e-3 | 0.40 |

  Five of the seven end within 13% of jaxpi's run. The other two follow it to 50k steps, then part from
  round-off and end below it.
- **The paper's batch sequence is an ordinary one.** jaxpi's sampler uses one fixed key whatever the seed, so
  every jaxpi run trains on the paper's sequence, while ours each draw their own. From seed 42's network
  that sequence leaves the early plateau sooner than other sequences, by up to 5k steps. But from seeds 0–5,
  jaxpi on other sequences ends where it does on the paper's (1.02× on geometric average, lower in 3 of 6),
  and leaves the plateau no later.
- **Our own batches end where jaxpi's do.** From jaxpi's seed 0–5 networks, our trainer ran on two fresh
  streams of our own sampler per network, in two rounds:

  | jaxpi's network | jaxpi, paper's batches | jaxpi, other batches | Ours, paper's batches | Ours, own batches (1) | Ours, own batches (2) |
  |---|--:|--:|--:|--:|--:|
  | seed 0 | 1.51e-3 | 1.80e-3 | 1.53e-3 | 2.58e-3 | 1.12e-3 |
  | seed 1 | 1.39e-3 | 2.16e-3 | 1.09e-3 | 2.06e-3 | 6.17e-4 |
  | seed 2 | 1.73e-3 | 1.60e-3 | 1.77e-3 | 1.76e-3 | 1.04e-3 |
  | seed 3 | 2.89e-3 | 2.31e-3 | 3.12e-3 | 3.26e-3 | 2.21e-3 |
  | seed 4 | 2.03e-3 | 2.90e-3 | 2.29e-3 | 2.52e-3 | 1.93e-3 |
  | seed 5 | 4.05e-3 | 2.39e-3 | 1.61e-3 | 4.96e-3 | 3.71e-3 |

  - The first round ended above the three runs on jaxpi's batches (their geometric mean) in every row, by
    1.34× on geometric average (Wilcoxon p = 0.03). The second ended below them in five of six, at 0.74×
    (p = 0.22), and left the early plateau no later than they did.
  - Pooled, the 12 runs on our batches and the 18 on jaxpi's have the same geometric mean, 2.01e-3
    (Mann–Whitney p = 0.69).
  - The two rounds differ from each other by 0.56×, in all six rows (p = 0.03), from the same networks and
    the same sampler. So six streams cannot resolve a factor of 1.3, and the first round's gap was chance.
  - This agrees with the batches themselves. Ours and jaxpi's are statistically the same: uniform in t and x,
    with t and x independent and no overlap between draws. Our trainer handles both through the same code.
- **Our own networks end where jaxpi's do.** Layer by layer, our initial weights match jaxpi's in scale
  (std / Glorot 0.92–1.02) and in their RWF scales (log g 0.49–0.51 ± 0.10). Over ten seeds, our `default`
  runs have a geometric mean of 2.23e-3 against 2.13e-3 for jaxpi's twelve (1.05×, Mann–Whitney p = 0.34),
  and a median of 2.60e-3 against 2.09e-3.
  - Our first five seeds had ended 1.4× above jaxpi's by median (p = 0.06). The next five ended at a median
    of 1.72e-3, so that gap was the spread of five runs.
  - Ours spread wider, from 3.6e-4 to 4.8e-3, against jaxpi's 1.4–4.1e-3. But jaxpi's twelve come from only
    six networks.

## Where the error comes from

In every run of `default`, `no_causal`, `no_rwf` and `no_grad_norm` checked, from both trainers, the error
sits at x = 0 late in time: over 99% of the squared error at t = 1 lies within |x| < 0.1. The initial condition u₀ = x² cos(πx) has a double zero at x = 0, so
the solution starts at the unstable state u = 0 there. Any error grows like e^{5t}, while the PDE residual
barely registers it. So the final error is set by how well a run holds u ≈ 0 near x = 0 early on, amplified
about 100×. That is why runs with the same losses can end several times apart.
`scripts/error_location.py` reports this for any finished run.

## Notes

- Hyperparameters and design choices not fixed by the papers were taken from jaxpi's configs and code. Each
  one is recorded, with its source, in [`decisions.md`](decisions.md). This repository contains no jaxpi
  code. The comparison runs of jaxpi, and the scripts that feed its exported networks and batches to our
  trainer, live outside it.
- The working log behind these numbers, with every run and test, is [`handoff.md`](../handoff.md).
