# Results: the Expert's Guide Allen–Cahn ablation (Table 2)

Relative L2 error after 200k steps, for each row of Table 2 in the Expert's Guide (arXiv:2308.08468). Each
row turns off one component of the training pipeline. All runs are float32, with TF32 off, on one RTX 3070 Ti
Laptop GPU, at about 1 h per run.

## Summary

- **Given the paper's starting network and batches, our trainer lands on the paper's values.** Four rows end
  within 0.89–1.01× of the paper. Of the other two, `no_fourier` and `no_causal`, each run can end in one of
  two ways, and the paper's single run and ours went different ways.
- **From our own seeds, `default` ends 5× above the paper, because the paper's run was lucky.** The paper's
  values are single runs from seed 42. jaxpi, the authors' code, run on seeds 0–5, ends 3.2× above the paper
  on `default` by median. jaxpi's runs also all train on one batch sequence, which turns out to be favorable,
  and that accounts for the remaining 1.5×. Given the same starting network and batches, our trainer ends
  where jaxpi's does.
- By median, our rows rank as the paper's do, except that `no_rwf` and `no_grad_norm` swap; they are within
  15% of each other in the paper.

## Our runs, from our own seeds

| Row | Fourier features | RWF | Grad norm | Causal | Seeds 0, 1, 2 (3, 4) | Median | Paper | Median / paper |
|---|:-:|:-:|:-:|:-:|---|--:|--:|--:|
| `default` | yes | yes | yes | yes | 2.93e-3, 2.27e-3, 2.27e-3 (3.67e-3, 4.81e-3) | 2.93e-3 | 5.84e-4 | 5.0 |
| `no_causal` | yes | yes | yes | no | 5.61e-3, 4.95e-3, 5.16e-3 | 5.16e-3 | 1.59e-3 | 3.2 |
| `no_rwf` | yes | no | yes | yes | 1.39e-2, 1.20e-2, 7.60e-3 | 1.20e-2 | 6.62e-3 | 1.8 |
| `no_grad_norm` | yes | yes | no | yes | 2.86e-3, 6.41e-3, 6.14e-3 | 6.14e-3 | 7.51e-3 | 0.82 |
| `no_fourier` | no | yes | yes | yes | 5.12e-1, 5.06e-1, 4.24e-1 | 5.06e-1 | 4.35e-1 | 1.2 |
| `plain` | no | no | no | no | 5.25e-1, 4.94e-1, 5.08e-1 | 5.08e-1 | 5.17e-1 | 0.98 |

`scripts/run_ablation.py` produces these runs with the configs in `configs/allen_cahn/ablation/`.

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
| `no_grad_norm` | 7.51e-3 | | 6.66e-3 | 0.89 |
| `no_fourier` | 4.35e-1 | | 7.37e-2 | 0.17 |
| `plain` | 5.17e-1 | | 4.85e-1 | 0.94 |

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
  paper's run does, or leaves it for the right one (0.05–0.08). 3 of our 8 runs of this row leave it,
  including this replay.

## Why our own seeds land above the paper

- **The paper's seed was favorable.** jaxpi's own `default` config, run unmodified, reproduces the paper from
  seed 42 (5.52e-4). Seeds 0–5 give 1.39–4.05e-3 instead, with a median of 1.9e-3, 3.2× the paper.
- **The paper's batch sequence was favorable too.** Every jaxpi run trains on it, whatever the seed, and
  ours each draw their own.
  - From seed 42's network, jaxpi with other sampler keys leaves the early plateau later than with the
    paper's key, by up to 5k steps. That is about when runs on our own batches leave it.
  - It helps to the end. From jaxpi's seed 0–5 networks, our trainer ends lower on the paper's batches than
    on its own in 5 of 6 runs: by 1.5× on geometric average.
  - Our own-seed `default` median (2.93e-3) is 1.56× jaxpi's seed 0–5 median, so seed and batches together
    account for the 5.0×.
- **Nothing measurable remains between our trainer and jaxpi's.** On the same network and batch, the two
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
  round-off and end below it. Each on its own batches, ours ended 1.2–1.3× above jaxpi's on geometric
  average.

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
