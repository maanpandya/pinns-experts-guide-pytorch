# Handoff (updated 2026-11-02)

The Windows laptop (RTX 3070 Ti Laptop GPU) is now the only workspace. Earlier work was done on a Mac, and there is nothing to send back there.

## Goal
Reproduce the Allen–Cahn ablation from the Expert's Guide (arXiv:2308.08468, **Table 2**) in PyTorch. The target is the relative L2 error of each row; the paper's values are in the table at the end of `windows_run_guide.md` (full pipeline: 5.84e-4). The implementation is independent and MIT-licensed (see `CLAUDE.md`).

## Where things stand
- **Code:** the whole Table 2 pipeline is done. 48 tests pass.
- **Laptop:** set up per `windows_run_guide.md`: Python 3.12, torch 2.14.1+cu132, a CUDA venv in `.venv`. The first GPU run found two platform problems, both fixed:
  - resume failed on CUDA because the RNG states were loaded onto the GPU (`c079158`);
  - Windows background power throttling (EcoQoS) slowed minimized runs 2–3× once the display turned off. `Trainer` now opts its own process out (`53a52b5`).
- **Derivative mode:** `--set training.derivatives=autograd`. It measured 11.1 ms/step against 21.4 for `func`, and gives the same values up to round-off.
- **RWF trains exp(s) itself since 2026-10-30** (`model.rwf.scale_param: direct`, jaxpi's version; D21). Runs before that trained s (`log`, EG Algorithm 2). The switch made no consistent difference to the error.
- **Results** (next section): every row has seeds 0–2, and `default` also seeds 3–4. By median, `no_grad_norm`, `no_fourier` and `plain` match the paper, and `no_rwf` is 1.8× high. `default` (3.9–8.2× high over five seeds) and `no_causal` (3.1–3.5×) are further off.
  - The paper's own logged runs (W&B) show the same loss weights and losses as ours, but lower error.
  - **Most of the `default` gap is the paper's seed** (see "jaxpi itself"). The paper ran seed 42. jaxpi itself on seeds 0–5 gives 1.39–4.05e-3 on the paper's batch sequence and 1.60–2.90e-3 on others; the median of those 12 runs is 2.1e-3, 3.6× the paper. From the paper's own starting network, our trainer reaches 3.44e-4.
  - **About 1.3× is left, and it is open.** From the same networks, our runs on our own batches end about 1.3× above every run on jaxpi's batches, from either trainer (6 of 6, p = 0.03). On jaxpi's batches, our trainer ends where jaxpi's does. Within jaxpi, the paper's batch sequence is no better than others, and our batches are statistically the same as jaxpi's. So this is chance or a difference not yet found ("Within jaxpi, the paper's batches are not favorable"). A second round is running.
  - **Given the same network and jaxpi's batches, our trainer follows jaxpi's run** step for step through 30k steps ("Ours leaves the early plateau a little later"). Given the paper's network and batch sequence ("The paper's runs, replayed in our trainer"), our `default`, `no_rwf`, `no_grad_norm` and `plain` runs end at 0.95, 1.01, 0.89 and 0.94× the paper. `no_causal` ends 3.3× high after falling back to the plateau at 19.8k. That was chance: from this network jaxpi falls back in 1 of 10 runs and ours in 3 of 11 (one-sided Fisher p = 0.33). `no_fourier` escapes the wrong solution the paper's run stays on (7.4e-2). jaxpi's own run on the same inputs escapes too (4.7e-2), and so do 4 of jaxpi's 7 runs of that row and 8 of our 14, to the same levels.
  - **From the paper's starting networks, all six rows land near the paper** ("Table 2 from the paper's starting networks"): four within 0.59–1.05×, `no_causal` 2.8× high, and `no_fourier` escaped the wrong solution (8.1e-2), which 2 of our 7 runs of that row do.
  - The error comes almost entirely from near x = 0, where u₀ has a double zero, so errors grow like e^{5t}.

## Current results (`scale_param: direct`)
| Row | Seeds 0, 1, 2 (3, 4) | Median | Paper | Median / paper |
|---|---|--:|--:|--:|
| default | 2.93e-3, 2.27e-3, 2.27e-3 (3.67e-3, 4.81e-3) | 2.93e-3 | 5.84e-4 | 5.0 |
| no_causal | 5.61e-3, 4.95e-3, 5.16e-3 | 5.16e-3 | 1.59e-3 | 3.2 |
| no_rwf | 1.39e-2, 1.20e-2, 7.60e-3 | 1.20e-2 | 6.62e-3 | 1.8 |
| no_grad_norm | 2.86e-3, 6.41e-3, 6.14e-3 | 6.14e-3 | 7.51e-3 | 0.82 |
| no_fourier | 5.12e-1, 5.06e-1, 4.24e-1 | 5.06e-1 | 4.35e-1 | 1.2 |
| plain | 5.25e-1, 4.94e-1, 5.08e-1 | 5.08e-1 | 5.17e-1 | 0.98 |

- **Folders:** the RWF rows are in `runs/allen_cahn/rwf_direct_seedN/`. `no_rwf` and `plain` have no RWF, so their runs in `ablation/` (seed 0), `ablation_seed1/` and `ablation_seed2/` count unchanged. Each run folder has `metrics.csv`, `summary.json`, `training.png`, `solution.png` and its checkpoint.
- **Ordering:** `default` < `no_causal` < `no_grad_norm` < `no_rwf` < `no_fourier` ≈ `plain`. The paper has `no_rwf` before `no_grad_norm`, within 15% of each other, and `no_fourier` 16% below `plain`; ours end within 0.4%.
- **The `log` runs** in the next two sections give the same picture. Pair by pair, the two versions differ about as much as two seeds do (D21).
- **The older folders hold `log` runs.** `runs/allen_cahn/ablation/` and `ablation_seed1/`–`ablation_seed4/` trained s. `scripts/run_ablation.py` now stops on their RWF rows rather than report them under the new default. To resume or extend them, add `--set model.rwf.scale_param=log`. Put new `direct` runs in their own `--out-root`.

## Seed-0 results (`log`)
| Row | Ours | Paper | Ours / paper | Wall time (min) |
|---|--:|--:|--:|--:|
| default | 3.39e-3 | 5.84e-4 | 5.8 | 91.9 (about 50 without throttling) |
| no_fourier | 5.33e-2 | 4.35e-1 | 0.12 | 43.9 |
| no_rwf | 1.39e-2 | 6.62e-3 | 2.1 | 52.1 |
| no_grad_norm | 7.28e-4 | 7.51e-3 | 0.10 | 52.0 |
| no_causal | 8.63e-3 | 1.59e-3 | 5.4 | 52.0 |
| plain | 5.25e-1 | 5.17e-1 | 1.0 | 39.5 |

- With seed 0 alone, `no_grad_norm` beat `default` by 4.6×, the reverse of the paper. Seeds 1–2 show this was seed noise; see the next section.
  - In `default`, λ_ic stays around 20–50 and λ_r around 0.5. The IC loss falls to about 1e-10, while the residual loss plateaus near 1e-7 from about 120k steps.
  - In `no_grad_norm`, both losses settle near 1–5e-8.
- `no_fourier` does about 8× better than the paper, which seeds 1–2 also show to be seed noise. `plain` matches the paper.
- In every row with causal weighting, the minimum causal weight rises to 1 and stays there (≥ 0.99). That happens from about 34–40k steps in `default`, `no_rwf` and `no_grad_norm`, and from 67k in `no_fourier`.
- **Outputs:** in `runs/allen_cahn/ablation/` (gitignored). That holds `ablation_table.md`, `ablation_rel_l2.png`, `run_notes.md`, and per row `metrics.csv`, `training.png`, `solution.png` and the checkpoints. `ablation_results.zip` in the repo root packs the same outputs without checkpoints.
- **Re-running:** to re-run or extend a row, use `windows_run_guide.md` step 7, adding `model.rwf.scale_param=log` to the `--set` list for this folder. Finished rows are skipped and an interrupted row resumes. Never pass `--no-resume` on `runs/allen_cahn/ablation/`.

## Seeds 1–2 and the remaining gap (2026-10-29, `log`)
- **Runs:** seeds 1 and 2 of every row, one process at a time, go into `runs/allen_cahn/ablation_seed1/` and `ablation_seed2/` (gitignored). `default` seeds 3 and 4 go into `ablation_seed3/` and `ablation_seed4/`.
  - `runs\seeds_chain.cmd` ran `default` and `no_grad_norm`. A Windows Update reboot at 09:34 on 2026-10-29 stopped it.
  - `runs\seeds_chain2.cmd` ran the rest, finishing at 23:12 on 2026-10-29. Re-running a chain is safe, since finished rows are skipped and interrupted rows resume. Nothing restarts a chain after a reboot.

| Row | Fourier | RWF | Grad norm | Seeds 0, 1, 2 (3, 4) | Median | Paper | Median / paper |
|---|:-:|:-:|:-:|---|--:|--:|--:|
| default | yes | yes | yes | 3.39e-3, 2.88e-3, 2.56e-3 (5.03e-3, 4.17e-3) | 3.39e-3 | 5.84e-4 | 5.8 |
| no_causal | yes | yes | yes | 8.63e-3, 2.66e-3, 4.89e-3 | 4.89e-3 | 1.59e-3 | 3.1 |
| no_rwf | yes | no | yes | 1.39e-2, 1.20e-2, 7.60e-3 | 1.20e-2 | 6.62e-3 | 1.8 |
| no_grad_norm | yes | yes | no | 7.28e-4, 5.65e-3, 5.70e-3 | 5.65e-3 | 7.51e-3 | 0.75 |
| no_fourier | no | yes | yes | 5.33e-2, 5.09e-1, 4.08e-1 | 4.08e-1 | 4.35e-1 | 0.94 |
| plain | no | no | no | 5.25e-1, 4.94e-1, 5.08e-1 | 5.08e-1 | 5.17e-1 | 0.98 |

- Per-seed tables and plots are in each root's `ablation_table.md` and `ablation_rel_l2.png`, and each run's `training.png` and `solution.png`.
- **Both seed-0 surprises were seed noise.**
  - `no_grad_norm` seed 0 (7.28e-4, which beat `default`) was an outlier. Seeds 1–2 follow the paper's `no_grad_norm` curve in EG Fig. 4 closely: a stall near 0.4–0.5 until 30–40k steps, a sharp drop, then 8.2e-3 at 100k against the paper's ~9.5e-3.
  - `no_fourier` seed 0 (5.33e-2) escaped the wrong solution at 60–80k steps. Seeds 1–2 stay at an error of 0.4–0.5 to the end, as in the paper, and so does seed 0 under D21's `direct` (0.51). Their L_r stays near 1.5–4e-4, so the causal weights sit at about 0.98–0.995 throughout, and causal weighting never holds the later times back.
- **By the median, the paper's ordering holds** (`default` < `no_causal` < `no_grad_norm`, `no_rwf` < `no_fourier` < `plain`). The one exception is that `no_rwf` and `no_grad_norm` swap; they are within 15% of each other in the paper. Each component helps, but by less than in the paper: disabling grad norm costs 1.7×, against the paper's 13×.
- **The remaining gap is in the rows with both RWF and grad norm.** Compared with the paper's own logged runs (JW in `docs/decisions.md`):
  - `default` is at 1.3–1.6× the paper's error at 10k steps (0.47–0.59 against 0.37), and 2.0–2.6× at 40k. Between 40k and 100k the paper's error drops 7.7×, ours 3.0–3.9×. Ours ends 4.4–8.6× high, so even the best of five seeds is far from the paper's run.
  - `no_causal` behaves the same way, more weakly (1.7–5.4× high).
  - `no_rwf` (grad norm without RWF) is 1.1–2.1× high, and seed 2 tracks the paper's curve from 20k steps on. `no_grad_norm` (RWF without grad norm) matches.
  - All rows share the learning-rate schedule, so the schedule is not the cause.

## The paper's own runs (2026-10-30)
jaxpi's Allen–Cahn README links a public Weights & Biases project, `jaxpi/allen_cahn`. It holds the paper's runs from 2023-04-14, one per Table 2 row, each logged every 100 steps. `docs/decisions.md` lists them as source JW. In everything the training sees, our `default` runs (the five `log` runs) look like the paper's:

| `default` at step | 10k | 40k | 100k | 200k |
|---|--:|--:|--:|--:|
| λ_r, paper | 0.67 | 0.52 | 0.51 | 0.51 |
| λ_r, ours (5 seeds) | 0.67 | 0.52 | 0.51–0.52 | 0.50–0.51 |
| λ_ic, paper | 33 | 21 | 21 | 25 |
| λ_ic, ours | 29–42 | 18–25 | 18–35 | 51–113 |
| L_r, paper | 1.2e-2 | 5.7e-5 | 4.7e-7 | 1.5e-7 |
| L_r, ours | 0.6–1.5e-2 | 0.5–16e-5 | 2.3–6.7e-7 | 0.8–4.4e-7 |
| min causal weight, paper | 0.64 | 1.00 | 1.00 | 1.00 |
| min causal weight, ours | 0.52–0.82 | 1.00 | 1.00 | 1.00 |
| rel. L2, paper | 0.37 | 7.6e-3 | 9.9e-4 | 5.84e-4 |
| rel. L2, ours | 0.47–0.59 | 1.5–2.0e-2 | 4.1–6.6e-3 | 2.6–5.0e-3 |

- **Same losses, higher error.** Our λ_r stays within 0.008 of the paper's at all 2,000 logged steps (four seeds; 0.03 for the fifth). Our losses reach the same levels, and the causal weights open up at the same time. Only the error differs.
  - Over the last 5k steps, the paper's `default` has rel. L2 / √L_r = 1.5. Our ten `default` runs (five regular, five D21) have 7.2–12.5.
  - The same holds, less strongly, in every row: `no_causal` 3.8 against 8.0–12.8, `no_rwf` 9.4 against 13.4–15.1, `no_grad_norm` 35 against 44–51.
  - So our networks reach the paper's loss levels but sit farther from the true solution.
- **That ratio is seed-dependent.** `no_grad_norm` seed 0 has 5.5, against 44–51 for seeds 1–2, at the same L_r. Its D21 twin, from the same network and batches, has 25 (and 2.9e-3 error instead of 7.3e-4). How close a run gets at a given loss depends on which solution it settles into, and small changes to the updates can change that.
  - Nothing the losses or weights show distinguishes the paper's `default` run from ours.
  - The paper's rows share one seed (D18): `default` and `default_ntk` take the identical first step. So the two runs agreeing near 5.7e-4 is one seed, not two.
- **Confirmed by JW:** D13 (the update timing, with the same jump at step 0), D14 (the plain row is 0.5174), D20 (the mean: the paper's λ_r settles at 0.51, while the sum would keep it above 1), and D2 (200k steps).
- **Hardware:** each paper run used one NVIDIA RTX A6000, with batch 4096 per run.
- **Ruled out:**
  - Settings whose effect would show in the logged losses or weights (D10–D13, D15, D20). These follow the paper's run.
  - The RWF parametrization (D21). jaxpi's version shows no consistent effect over 14 paired runs (below).
  - IC over-weighting as such. At equal residual loss (L_r ≈ 6e-7), `default` has about half the error of `no_grad_norm`.
  - The reference solution and metric. jaxpi's own reference data (Chebfun) and ours agree to 1.5e-5 relative L2 (D17).
  - Changes in jaxpi since 1.0.0. Its library and Allen–Cahn example have changed only cosmetically since 2023-08-06 (`docs/decisions.md`, Sources). The paper's runs predate 1.0.0, though, so differences in their pre-release code cannot be ruled out.
- **Matmul precision: not the cause** (tested with jaxpi itself, below).
  - jaxpi has pinned JAX's float32 matmul precision to `"highest"` only since 2024-11 (`df13b96a`). That came after a report of hardware-dependent results (jaxpi issue #22), so the paper's runs used JAX's default.
  - jaxpi's seed 42 at JAX's default precision (TF32 on this GPU) ends at 7.31e-4, against 5.52e-4 at full precision. Our own runs keep TF32 off (rule 1 of `windows_run_guide.md`).
- **D21 test (`runs\rwf_direct_chain.cmd` and `runs\rwf_direct_chain2.cmd`, `model.rwf.scale_param=direct`, into `runs/allen_cahn/rwf_direct_seed*/`):**
  - Each pair starts from the same network and sees the same batches. The 11 pairs of `default`, `no_causal` and `no_grad_norm` are within 7% at 10k steps, then drift apart about as far as different seeds do.
  - `default`, seeds 0–4: `direct` is lower in 4 of 5 (ratios 0.73–0.89, and 1.15 for seed 4). Median 3.39 → 2.93e-3, still 5.0× the paper.
  - `no_causal` and `no_grad_norm`, seeds 0–2: `direct` is higher in 5 of 6 (ratios 0.65–3.9). `no_grad_norm` seed 0 loses its lucky 7.3e-4 and ends at 2.9e-3.
  - `no_fourier`, seeds 0–2 (`runs\rwf_direct_chain3.cmd`): seeds 1–2 within 4%. Seed 0 loses its lucky 5.3e-2 and ends at 0.51.
  - Over all 14 pairs `direct` is lower in 6, and the geometric mean of the ratios is 1.28 (0.98 without the two seed-0 outliers). The table is in D21.
  - Medians, with `no_rwf` and `plain` unchanged since they have no RWF:

    | Row | Seeds | `log` | `direct` | Paper |
    |---|---|--:|--:|--:|
    | default | 0–4 | 3.39e-3 | 2.93e-3 | 5.84e-4 |
    | no_causal | 0–2 | 4.89e-3 | 5.16e-3 | 1.59e-3 |
    | no_grad_norm | 0–2 | 5.65e-3 | 6.14e-3 | 7.51e-3 |
    | no_fourier | 0–2 | 4.08e-1 | 5.06e-1 | 4.35e-1 |

    The row ordering is the same under both, though under `direct` `no_fourier` ends within 0.4% of `plain` (5.08e-1).
- **λ_ic climbs late in every `default` seed,** to 51–113 at 200k, where the paper's stays at 25. L_ic falls to 1e-10–2e-9. By then the learning rate is below 1e-6, so this barely moves the parameters.

## jaxpi itself (2026-10-31)
jaxpi's own Allen–Cahn `default` config, run unmodified on this laptop (source JR in `docs/decisions.md`). This separates "the paper's seed was favorable" from "our implementation differs somewhere the logs don't show".

| jaxpi run | rel. L2 at 10k / 40k / 100k | Final rel. L2 | rel. L2 / √L_r |
|---|---|--:|--:|
| paper (W&B) | 0.37 / 7.6e-3 / 9.9e-4 | 5.84e-4 | 1.5 |
| seed 42, full precision | 0.38 / 7.7e-3 / 9.4e-4 | 5.52e-4 | 2.4 |
| seed 0, full precision | 0.54 / 1.1e-2 / 2.5e-3 | 1.51e-3 | 5.6 |
| seed 1, full precision | 0.55 / 1.1e-2 / 2.4e-3 | 1.39e-3 | 4.3 |
| seed 2, full precision | 0.53 / 1.4e-2 / 3.2e-3 | 1.73e-3 | 6.8 |
| seed 3, full precision | 0.51 / 9.4e-3 / 3.8e-3 | 2.89e-3 | 10.5 |
| seed 4, full precision | 0.49 / 8.2e-3 / 2.9e-3 | 2.03e-3 | 5.4 |
| seed 5, full precision | 0.46 / 1.4e-2 / 5.3e-3 | 4.05e-3 | 10.6 |
| seed 42, JAX default precision (TF32) | 0.37 / 7.2e-3 / 9.6e-4 | 7.31e-4 | 1.8 |
| ours, `default`, 10 runs | 0.47–0.59 / 1.4–2.1e-2 / 3.6–6.6e-3 | 2.3–5.0e-3 | 7.2–12.5 |

- **The paper used seed 42**, and jaxpi reproduces the paper from it. With JAX's pre-0.5 random streams, jaxpi's seed 42 logs the paper's first step, so it starts from the same network (D18). It ends at 5.52e-4, and its error stays within about 10% of the paper's run the whole way, although its losses drift up to 2× from the paper's.
- **Seed 42 was favorable.** jaxpi's seeds 0–5 end at 1.39–4.05e-3, 2.5–7.3× above it. Their median, 1.9e-3, is 3.2× the paper's 5.84e-4. At 10k steps they are at 0.46–0.55, inside our range.
- **Ours are about 1.5× worse than jaxpi's typical seed, and that may be chance.** The median of our 16 `default` runs that did not start from seed 42's network is 2.9e-3, against jaxpi's 1.9e-3. Same-network pairs (below) put ours worse in 5 of 7, by 1.2× on geometric average, but single pairs range from 0.47× to 3.1×. So most of the gap to the paper is the favorable seed: 3.2×, or 3.6× over jaxpi's 12 runs on two batch sequences. What is left between our runs and jaxpi's goes with the batches, not the trainer, and is still open (below).
- **Not the initialization:** jaxpi's initial weights and ours have the same statistics, per layer: Glorot spread, the truncation at 2σ, log g ≈ N(0.5, 0.1²), and B ~ N(0, 1).
- **Not the training step:** from jaxpi's seed-42 weights and one fixed batch, our `Trainer.train_step` and jaxpi's step agree.
  - Both losses are identical to 8 digits, and the chunk losses and causal weights agree to 3e-7.
  - The raw grad-norm weights are identical to 6 digits, and λ_ic after the update is 67.7440 against 67.7431.
  - After one Adam step, every parameter moves in the same direction, within 1.5e-5 on steps of 1e-3.
  - jaxpi's optimizer is plain Adam with the same exponential decay as ours, with no warmup, clipping or gradient accumulation.
  - The one remaining difference is the initial-condition grid. jaxpi uses its data's `linspace(-1, 1, 512)`, which holds x = ±1 twice; ours uses 512 periodic points on [-1, 1). The step test used jaxpi's grid.
- **λ_ic climbs late in jaxpi too** (36–67 at 200k, against the paper's 25), so that climb in our runs is not a difference from jaxpi.
- **TF32 is not the cause.** At JAX's default precision, seed 42 follows the paper's curve as closely as at full precision, and ends 25% higher (7.31e-4). This does not show which precision the paper used.
- **Same starting network, two trainers.** Our trainer loads jaxpi's initial weights, then draws its own batches:

  | jaxpi's initial network | jaxpi's trainer | Ours | Ours / jaxpi's |
  |---|--:|--:|--:|
  | seed 42 | 5.52e-4 | 3.44e-4 | 0.62 |
  | seed 0 | 1.51e-3 | 2.27e-3 | 1.50 |
  | seed 1 | 1.39e-3 | 4.35e-3 | 3.13 |
  | seed 2 | 1.73e-3 | 2.59e-3 | 1.50 |
  | seed 3 | 2.89e-3 | 3.34e-3 | 1.16 |
  | seed 4 | 2.03e-3 | 2.87e-3 | 1.41 |
  | seed 5 | 4.05e-3 | 1.90e-3 | 0.47 |

  - **jaxpi's initial networks are no better than ours.** From jaxpi's seeds 0–5, our trainer ends at 1.90–4.35e-3 (median 2.73e-3), the same spread as from our own networks (2.27–4.81e-3, median 2.93e-3).
  - **jaxpi's trainer is at most slightly better.** Over seeds 0–5 it gives 1.39–4.05e-3 (median 1.9e-3), against ours 1.90–4.35e-3 (median 2.7e-3) from the same networks. Ours is worse in 5 of the 7 pairs; the geometric mean of the ratios is 1.19 (1.32 without seed 42). Ours is behind at 40k steps in 6 of 7 pairs (0.82–1.43× jaxpi's error). On jaxpi's batches the difference disappears ("On jaxpi's batches, the 1.3× goes away" below).
  - The final losses of each pair are close (L_r within 1.8×, L_ic within 3.4×, in both directions).
  - The only known difference left between the loops is the IC grid (above). **It makes no difference (D11).** **IC-grid test** (`runs/allen_cahn/from_jaxpi_init_icgrid/seedN/`): the same network and batches as `from_jaxpi_init/seedN/`, with the IC loss on jaxpi's `linspace(-1, 1, 512)` instead of our periodic grid:

    | jaxpi's network | jaxpi's trainer | Ours, our grid | Ours, jaxpi's grid | jaxpi's grid / our grid |
    |---|--:|--:|--:|--:|
    | seed 0 | 1.51e-3 | 2.27e-3 | 2.58e-3 | 1.14 |
    | seed 1 | 1.39e-3 | 4.35e-3 | 2.06e-3 | 0.47 |
    | seed 2 | 1.73e-3 | 2.59e-3 | 1.76e-3 | 0.68 |
    | seed 3 | 2.89e-3 | 3.34e-3 | 3.26e-3 | 0.97 |
    | seed 4 | 2.03e-3 | 2.87e-3 | 2.52e-3 | 0.88 |
    | seed 5 | 4.05e-3 | 1.90e-3 | 4.96e-3 | 2.61 |

    - Each pair stays within 10% through 40k steps, then drifts apart, as same-network pairs do. The error sits at x = 0 in all six (0.065–0.071 × the error at (t = 1, x = 0)). The error at (t = 0, x = 0) is about the same in each pair (2.3–3.2e-5 with jaxpi's grid, 2.7–6.4e-5 with ours), although our grid has a point at x = 0 and jaxpi's does not.
    - Over the six pairs, jaxpi's grid / ours has a geometric mean of 0.97 (Wilcoxon p = 0.84). After seeds 0–2 it looked like 0.71, which seeds 3–5 undid.
    - On either grid, ours ends about 1.3× above jaxpi's run from the same network (geometric mean 1.32 with our grid, 1.28 with jaxpi's). jaxpi's runs all train on the paper's batch sequence (key 1234); ours each draw their own. On that sequence the 1.3× goes away (next bullet). But within jaxpi, that sequence is no better than others (the bullet after).
  - **On jaxpi's batches, the 1.3× goes away** (`runs/allen_cahn/paper_batches/seedN/`): our `default` from each of jaxpi's seed 0–5 networks, on jaxpi's batch sequence with jaxpi's IC grid, for 200k steps. Each continues the 30k-step replay in `batch_test/replay_seedN/`. The last column is the IC-grid test above (the same network and grid, our own batches):

    | jaxpi's network | jaxpi's trainer | Ours, jaxpi's batches | Ours / jaxpi's | Ours, own batches (jaxpi's grid) |
    |---|--:|--:|--:|--:|
    | seed 0 | 1.51e-3 | 1.53e-3 | 1.01 | 2.58e-3 |
    | seed 1 | 1.39e-3 | 1.09e-3 | 0.78 | 2.06e-3 |
    | seed 2 | 1.73e-3 | 1.77e-3 | 1.02 | 1.76e-3 |
    | seed 3 | 2.89e-3 | 3.12e-3 | 1.08 | 3.26e-3 |
    | seed 4 | 2.03e-3 | 2.29e-3 | 1.13 | 2.52e-3 |
    | seed 5 | 4.05e-3 | 1.61e-3 | 0.40 | 4.96e-3 |

    - Four of the six end within 13% of jaxpi's run, as seed 42's replay does (`paper_replay/default`, 1.01). The other two end below jaxpi's. Over seeds 0–5 the geometric mean of ours / jaxpi's is 0.85 (Wilcoxon p = 1.0), and the medians are 1.69e-3 against 1.88e-3.
    - Each pair stays within 15% at 10k, 20k, 30k and 50k steps. Then seed 5 parts between 50k and 100k (5.3e-3 for jaxpi's run at 100k, 2.8e-3 for ours), and seed 1 between 100k and 150k, as same-network pairs do.
    - On jaxpi's batches, our runs end lower than on their own in 5 of the 6 pairs, with the same grid (0.67× on geometric average, Wilcoxon p = 0.06). jaxpi's own runs on other sequences do not end higher (next bullet).
    - The error sits at x = 0 in all six (0.062–0.079 × the error at (t = 1, x = 0)).
    - **Reading:** given the same network and batches, our trainer ends where jaxpi's does. So the 1.3× between the trainers goes with the batches, not with the training step. This was first read as jaxpi's sequence being favorable, matching the 1.56× between our own-seed median (2.93e-3) and jaxpi's (1.88e-3). jaxpi itself does not bear that out (next bullet).
- **Within jaxpi, the paper's batches are not favorable** (`~/jaxpi-runs/chain7.sh`: jaxpi's `default` from seeds 0–5 on sampler key 100 + seed, 200k steps; the scratchpad's `keys_compare.py` and `batch_stats.py`):

  | jaxpi's network | jaxpi, key 1234 | jaxpi, key 100 + seed | Ours, key 1234 | Ours, own batches |
  |---|--:|--:|--:|--:|
  | seed 0 | 1.51e-3 | 1.80e-3 | 1.53e-3 | 2.58e-3 |
  | seed 1 | 1.39e-3 | 2.16e-3 | 1.09e-3 | 2.06e-3 |
  | seed 2 | 1.73e-3 | 1.60e-3 | 1.77e-3 | 1.76e-3 |
  | seed 3 | 2.89e-3 | 2.31e-3 | 3.12e-3 | 3.26e-3 |
  | seed 4 | 2.03e-3 | 2.90e-3 | 2.29e-3 | 2.52e-3 |
  | seed 5 | 4.05e-3 | 2.39e-3 | 1.61e-3 | 4.96e-3 |

  (Ours are `paper_batches/` and `from_jaxpi_init_icgrid/`, both on jaxpi's IC grid.)
  - **jaxpi on other keys ends where it does on key 1234:** 1.02× on geometric average, higher in 3 of 6 (Wilcoxon p = 1.0). Nor does it leave the plateau later. Averaged over the six, it drops below 0.3, 0.1 and 0.03 at 13.7k, 15.2k and 24.5k steps, against 12.8k, 15.5k and 23.8k on key 1234. So from seeds 0–5 the paper's sequence is an ordinary one. Only from seed 42 (30k-step runs, "From seed 42's network, that sequence escapes early") did it escape early.
  - **Our runs on our own batches are the odd ones out.** Per network, they end above the mean (in logs) of the three runs on jaxpi's batches in 6 of 6, by 1.04–1.99× (geometric mean 1.34, Wilcoxon p = 0.03). Against jaxpi on key 100 + seed alone, the factor is 1.25× (higher in 4 of 6, p = 0.22).
    - They also leave the plateau later: on average at 13.8k, 16.8k and 26.0k, against 13.0k, 15.3k and 24.0k for ours on key 1234. At 40k steps they are above jaxpi on key 100 + seed in 5 of 6.
    - With our own-seed runs added, 11 runs on our own batches against jaxpi's 12 give Mann–Whitney p = 0.045, at 1.34× on geometric average. But our own-seed runs draw from the same six streams (seed s draws with generator seed s + 1), so they add new networks, not new batches.
  - Within one network, batches and round-off spread the final error by about 0.3 in log (1.35×) per run, in both trainers.
  - **Our batches are statistically the same as jaxpi's.** `sampler_check.py` compared 2000 batches each of our CUDA generator (seeds 0–2, as the trainer draws them) and jaxpi's key 1234:
    - t and x are uncorrelated within a batch: the sd of the correlation is 0.015–0.016, as for iid draws.
    - No values repeat between t and x, or between consecutive batches, beyond float32 coincidences.
    - t and x are uniform (KS p = 0.48–0.95) and fill a 16 × 16 grid evenly (χ² p = 0.61–0.99). Our CPU generator, which training does not use, gives KS p = 0.02 for x, which is about chance over the tests run.
    - 1.00% of the points fall in t ≤ 0.1, |x| < 0.1.
    - Both pair the sorted t with x at random: ours permutes x along with t, and jaxpi leaves x in place.
    - Our trainer takes the same code path for both batch sources; `from_jaxpi_init.py` swaps only `sample_batch`.
  - So nothing known separates our batches from jaxpi's, and chance is the likely reading. It is not settled at p = 0.03 with six streams.
  - **Second round, running** (`runs\own_batches2.ps1`). Per network, ours runs on a fresh stream (generator seed 101 + s via `--seed 100 + s`, jaxpi's IC grid, into `runs/allen_cahn/own_batches2/seedN/`). It alternates with jaxpi on key 200 + s (`~/jaxpi-runs/chain8.sh`).
- **The error comes from one spot: x = 0, early on** (`scripts/error_location.py` prints this for any finished run). It holds in all 17 `default` runs checked (five `log`, five `direct`, seven from jaxpi's networks), and in `no_causal`, `no_grad_norm` and `no_rwf`:
  - At t = 1, over 99% of the squared error lies within |x| < 0.1, and about 90% of all of it falls in t > 0.75.
  - u₀ = x² cos(πx) has a double zero at x = 0, so the solution starts at the unstable state u = 0 there, and errors grow roughly like e^{5t}. An error growing that way nearly satisfies the linearized PDE, so the residual hardly sees it. Only the initial condition holds it down.
  - The error at x = 0 is positive in every run. In `default` it is 6e-6 to 6.4e-5 at t = 0, 5–17× more by t = 0.1, and another 110–140× by t = 1. The final rel. L2 is 0.059–0.070 × the error at (t = 1, x = 0) in 16 of the 17 runs (0.10 from seed 42's network).
  - So each run's error is fixed by how closely the network holds u ≈ 0 near x = 0 for t ≤ 0.1, amplified about 100×. That is why runs with the same losses end up to 14× apart, and why rel. L2 / √L_r ranges from 1.5 to 12.5.
  - In `default` the early residual sets that error: it grows 5–17× by t = 0.1, faster than the e^{0.5} ≈ 1.65 a pure initial-condition error would. In `no_grad_norm` the error at t = 0 is 10× larger (3–6e-4) and grows only 1.6–1.7× by t = 0.1, so there the IC fit sets it. That is likely how grad-norm weighting helps here: by up-weighting the IC loss, it pins u at x = 0.
  - The IC error elsewhere peaks at the kink of u₀ at x = ±1 (up to 1e-3), but that region settles to u = −1 and contributes nothing.
  - **jaxpi's trainer fails the same way.** jaxpi's runs of seeds 2–5 saved their final networks (`~/jaxpi-runs/*_final.npz`). Loaded into our model, they match jaxpi's predictions to under 1e-6 and score the same on our grid and reference. In all four, 99.9% or more of the error at t = 1 lies within |x| < 0.1, the error at x = 0 is positive, and rel. L2 is 0.063–0.069 × the error at (t = 1, x = 0). Their error at (t = 0.1, x = 0) is 1.95–5.17e-4, against 2.87–3.97e-4 from our trainer on the same starts.
- **Table 2 from the paper's starting networks** (our trainer from jaxpi's seed-42 network for each row, `runs/allen_cahn/from_paper_init/<row>/`; `default` is the seed-42 run above). The last column is the median from "Current results", our runs from our own networks.

  | Row | Paper | jaxpi, seed 42 | Ours from the paper's network | Ours / paper | Ours, own networks (median) |
  |---|--:|--:|--:|--:|--:|
  | `default` | 5.84e-4 | 5.52e-4 | 3.44e-4 | 0.59 | 2.93e-3 |
  | `no_causal` | 1.59e-3 | 1.29e-3 | 4.44e-3 | 2.79 | 5.16e-3 |
  | `no_grad_norm` | 7.51e-3 | 7.72e-3 | 5.51e-3 | 0.73 | 6.14e-3 |
  | `no_rwf` | 6.62e-3 | 3.83e-3 | 6.96e-3 | 1.05 | 1.20e-2 |
  | `no_fourier` | 4.35e-1 | 4.74e-2 | 8.10e-2 | 0.19 | 5.06e-1 |
  | `plain` | 5.17e-1 | 5.03e-1 | 5.04e-1 | 0.97 | 5.08e-1 |

  - **jaxpi itself, seed 42, reproduces the paper on `no_causal` and `no_rwf` as on `default`.** `no_causal` tracks the paper's run within about 20% the whole way (8.45e-2 against 8.43e-2 at 20k) and ends 0.81× the paper's. `no_rwf` leaves the plateau earlier than the paper's run (8.3e-2 against 0.23 at 15k) and ends 0.58× the paper's.
  - `no_grad_norm` follows the paper's run closely the whole way: error, both losses, and the causal weights opening by 20k.
  - `no_rwf` leaves the plateau a little later (0.47 at 10k, against the paper's 0.16), then follows the paper's run within 15% from 20k to the end. Its own-network runs (7.6e-3 to 1.39e-2) are all above it, so here too the paper's seed was on the good side.
  - `no_causal` follows it to 10k, then leaves the plateau later (0.50 at 20k, against the paper's 0.084), and stays 1.5–2.8× behind to the end. jaxpi's own run from that network ends at 1.29e-3, so ours is 3.4× jaxpi's here, the widest same-network pair so far.
  - `no_fourier` follows the paper's run to 40k (0.51 against 0.52), then escapes the wrong solution between 40k and 100k, while the paper's run stays at 0.40–0.52. Our `log` seed 0 escaped the same way (5.33e-2; "Seeds 1–2 and the remaining gap" above), so this row has two outcomes: 2 of our 7 runs escape, to 0.05–0.08, and 5 stay at 0.41–0.51, like the paper's single run. Once escaped, its error also sits at x = 0.
  - `plain` follows the paper's run closely the whole way, and ends within 3% of it.
  - `no_causal`, `no_grad_norm` and `no_rwf` put their error at x = 0, like every other run that reaches the right solution: 99.9% or more of the squared error at t = 1 lies within |x| < 0.1, and rel. L2 is 0.064–0.075 × the error at (t = 1, x = 0). `plain` and the stuck `no_fourier` runs are wrong everywhere instead (about 5% of the error lies within |x| < 0.1).
  - **Reading:** from the paper's own networks, four of the six rows land within 0.59–1.05× of the paper, `no_causal` is 2.8× high, and `no_fourier` took the other of its two outcomes. Both of those fall within the run-to-run spread seen elsewhere (same-network pairs differ by up to 3.1×, and `no_fourier` escapes in 2 of 7 runs). So nothing here points to a systematic difference between our pipeline and the paper's.
  - `no_causal` and `no_grad_norm` start from `default`'s network (jaxpi's configs for them differ only in the switch, so they build the same one). `no_rwf`, `no_fourier` and `plain` use their own seed-42 networks, exported from jaxpi's configs of the same names. Each matches jaxpi's initial predictions to under 1e-6.
  - The queue's window closed at 17:59, with `no_grad_norm` at step 132k (the laptop was in use then). It was relaunched at 18:00 and resumed from the checkpoint.
- **Ours leaves the early plateau a little later than jaxpi. Given the same batches it does not, so this goes with the batches, not the trainer.**
  - Over the nine same-network pairs (`default` seeds 42 and 0–5, `no_causal` and `no_rwf` seed 42), read on the same 1000-step grid, ours first drops below rel. L2 0.3 and 0.1 later in 6 pairs and at the same step in 3, never earlier: about 8% (1–2k steps) later on average. Below 0.03 it is later in 7 of 9. (Reading jaxpi's 100-step log against our 1000-step evaluations exaggerates this, to 9 of 9.)
  - The IC grid is not the cause: from jaxpi's seed-0 network, our trainer with jaxpi's IC grid drops below 0.3, 0.1 and 0.03 at 17k, 18k and 25k, the same steps as with our grid.
  - **Given jaxpi's batches, our trainer follows jaxpi's trajectory.** jaxpi's residual sampler uses the fixed key `PRNGKey(1234)`, whatever the seed, so every jaxpi run (and every paper run) trains on the same batch sequence. Exported and fed to our trainer from seed 42's network, with jaxpi's IC grid, it gives λ_ic = 49.25 after the first update (the paper's 49.2; ours with our own batches gives 52), the same λ_ic to two decimals through 3k steps, and rel. L2 within a few percent of jaxpi's run through 30k steps: below 0.3, 0.1 and 0.03 at 10.6k, 14.1k and 23.4k, against jaxpi's 10.6k, 14.0k and 22.9k.
  - So the two training loops are the same, and the escape timing is set by the network and the batch sequence. All nine jaxpi runs share one sequence, so the pairs are not independent on jaxpi's side.
  - **From seed 42's network, that sequence escapes early** (30k-step runs in `runs/allen_cahn/batch_test/` and `~/jaxpi-runs/*_key*_30k.jsonl`; steps where rel. L2 first drops below 0.3, 0.1, 0.03):

    | Run | < 0.3 | < 0.1 | < 0.03 |
    |---|--:|--:|--:|
    | jaxpi, sampler key 1234 (the paper's batches) | 10.6k | 14.0k | 22.9k |
    | ours on those batches | 10.6k | 14.1k | 23.4k |
    | jaxpi, keys 1, 2, 3 | 11.6k, 11.4k, 11.5k | 14.4k, 14.4k, 14.7k | 23.9k, 25.2k, 27.7k |
    | ours, batch seeds 43, 44 | 11.6k, 10.8k | 15.0k, 14.0k | 25.6k, 22.7k |
    | ours, batch seed 42 (1000-step grid) | 12k | 16k | 25k |

    From this network, with other batch sequences, jaxpi leaves the plateau as late as ours does, and key 1234 is the earliest of the seven runs at 0.3. From seeds 0–5 it is not early. There, our runs on our own batches are the late ones ("Within jaxpi, the paper's batches are not favorable").
  - **The same holds from jaxpi's seed 0–5 networks.** On jaxpi's batches, our trainer drops below 0.3, 0.1 and 0.03 within 0.1–0.4k steps of jaxpi's own runs, and its error at 10k, 20k and 30k is within a few percent of theirs. The two exceptions are seed 2 at 30k (2.2e-2 against 3.1e-2), and seed 5 at 0.1. In seed 5 both runs reach about 0.12 by 12.4k and then fall back to about 0.5, jaxpi's at 12.8k and ours at 14k, after ours had dipped to 9.6e-2. Both recover by 17–18k.
- **The paper's runs, replayed in our trainer** (`runs/allen_cahn/paper_replay/<row>/`): for each row, the paper's network (jaxpi seed 42), the paper's batch sequence (jaxpi's sampler key 1234, exported to `jaxpi_batches_key1234_200k.npy`, 6.5 GB) and jaxpi's IC grid, all 200k steps. jaxpi with key 1234 matches the paper's logged steps 0 and 100 to 3–4 digits on `default`, `no_causal` and `no_rwf`, so these re-run the paper's own experiment in our code.

  | Row | Paper | jaxpi, same inputs | Ours, same inputs | Ours / paper |
  |---|--:|--:|--:|--:|
  | `default` | 5.84e-4 | 5.52e-4 | 5.56e-4 | 0.95 |
  | `no_causal` | 1.59e-3 | 1.29e-3 | 5.26e-3 | 3.31 |
  | `no_rwf` | 6.62e-3 | 3.83e-3 | 6.72e-3 | 1.01 |
  | `no_grad_norm` | 7.51e-3 | 7.72e-3 | 6.66e-3 | 0.89 |
  | `no_fourier` | 4.35e-1 | 4.74e-2 | 7.37e-2 | 0.17 |
  | `plain` | 5.17e-1 | 5.03e-1 | 4.85e-1 | 0.94 |

  - **`default` reproduces the paper.** Our replay stays within about 15% of jaxpi's run at every logged step (rel. L2 0.374 against 0.376 at 10k, 9.4e-4 against 9.4e-4 at 100k) and ends within 1% of it. jaxpi's run is itself within about 10% of the paper's. Its error sits at x = 0, as in every other run (0.102 × the error at (t = 1, x = 0)).
  - **`no_causal` falls back to the plateau where jaxpi's run does not.** Our replay follows jaxpi's run and the paper's to 19.7k (rel. L2 0.084 against 0.085). Then one burst in the losses at 19.8k (L_ic 3.6e-4 and L_r 1.2e-2, from about 2e-6 and 3e-4) throws it back to 0.50. It escapes again by 23k, 3k steps behind, and stays 1.5–4× behind to the end. Our run from the same network on our own batches fell back the same way at 19.3k and ended at 4.44e-3. jaxpi's and the paper's runs have bursts there too (L_r 1.1e-2 at 18k) without falling back. Both trainers fell back from seed 5's network (at 12.8k and 14k), so falling back is not ours alone. Nor are our loss bursts more frequent: over steps 5k–30k of the seven 30k replays on jaxpi's batches, 54 logged values of our L_r or L_ic exceed 10× the median of the previous 1k steps, against 64 of jaxpi's (9 against 15 at 30×).
  - **The fall-back is chance: jaxpi falls back from this network too.** `no_causal` from the paper's network, 30k steps, on other batch sequences, in two rounds (a fall-back is the error dropping below 0.15 and then climbing back above 0.3):

    | Runs | Fell back | Below 0.1 at (median) | Error at 30k (median) |
    |---|--:|--|--|
    | jaxpi, sampler keys 1234 and 1–9 | 1 of 10 (key 6: 0.074 to 0.59 at 21.3k, below 0.15 again at 26.1k) | 18.3–20.9k (19.1k) | 2.6–5.6e-2 (3.3e-2) |
    | ours, the paper's batches and batch seeds 42–51 | 3 of 11 (the paper's batches and seeds 42, 43, at 19.4–20.0k, peaks 0.50–0.63) | 18.5–20.3k (19.0k) | 2.8–5.5e-2 (3.8e-2) |

    - After a first round of 0 of 4 for jaxpi against 3 of 5 for ours, the second round gave 1 of 6 for jaxpi (plus key 9's brief rise to 0.30 at 21.6k) against 0 of 6 for ours. Overall, 3 of 11 against 1 of 10 has a one-sided Fisher p of 0.33.
    - jaxpi's fall-back looks like ours: a jump from below 0.08 to 0.59 within 100 steps, about 1–3k steps after the run first drops below 0.1, and a slow climb back over the next 3–5k steps.
    - Bursts are about as frequent in both. Over steps 5k–30k of all 21 runs, logged values of L_r or L_ic above 10× the median of the previous 1k steps number 53 per run in ours against 49 in jaxpi's (medians; Mann–Whitney p = 0.34). Above 30× ours has slightly more, 22 against 19 (p = 0.03). In the seven `default` replays ours had fewer (9 against 15 in total above 30×), and our runs with the most bursts (batch seeds 48–50) did not fall back. jaxpi's first-round runs have 5–8 bursts above 100× each, against our 2–8, mostly at 5–12k on the plateau, where they do no harm. What decides a fall-back is how fragile the run is when the burst comes, not the burst's size. On the paper's batches, jaxpi's run has bursts as large as the one that threw ours back (L_ic 1.2–2.3e-4 at 20.0k, 20.3k and 20.8k) and stays put.
    - The first loss weights on the paper's batches match jaxpi's to 5 digits (λ_ic 364.18 against 364.17 after the first update).
    - Without its fall-backs, our median at 30k is 3.4e-2 against jaxpi's 3.2e-2.
  - **`no_rwf` reproduces the paper.** Our replay ends within 1% of the paper (6.72e-3 against 6.63e-3) and is within 25% of it at every logged step from 30k on (3.3e-2 against 3.9e-2 at 30k, 9.3e-3 against 9.0e-3 at 100k). jaxpi's run on the same inputs ends 1.7× lower (3.83e-3). The paper's run has a burst at 15k (back up to 0.23 from 0.16) that neither jaxpi's nor ours has, so the paper's and jaxpi's runs also part from round-off.
  - **`no_grad_norm` reproduces the paper.** Our replay ends at 0.89× the paper (6.66e-3 against 7.51e-3). Its error is within 20% of the paper's at every logged step: both sit on the plateau to 20k (0.52 against 0.51), escape by 40k (2.3e-2 against 2.8e-2), and end with the same losses (L_ic 7.3e-8 against 9.6e-8). jaxpi's run on the same inputs (`~/jaxpi-runs/chain6.sh`) ends at 7.72e-3, 1.03× the paper. It leaves the plateau a little later than the paper's run and ours (below 0.1 from 37.2k, against 31.8k and 27.0k). (The paper's logged step 0, rel. L2 3.7 here as in `default`, comes after the first Adam update; ours comes before it. From step 100 on the two agree: 0.767 against 0.771.)
  - **`plain` reproduces the paper.** Our replay ends at 0.94× the paper (4.85e-1 against 5.17e-1). It follows the paper's run at every logged step: stuck at 0.99 to 20k, down to 0.58 against 0.57 at 40k, then 0.49–0.50 against 0.52. Both end on the wrong solution, as do all our `plain` runs. jaxpi's run on the same inputs ends between them, at 5.03e-1.
  - **`no_fourier` escapes the wrong solution where the paper's run does not.** Our replay leaves it from 44k (below 0.45; 0.3 at 64k, 0.1 at 102k) and ends at 7.37e-2. The paper's run never goes below 0.40. Unlike `default`, this row parts from the paper's run within the first 1k steps (λ_ic 1.55 against 1.28 after its first update at 1k; rel. L2 0.68 against 0.71 at 10k, 0.49 against 0.52 at 40k).
  - **That is chaos, not a difference in the code.** One update of `no_fourier` from the same network and batch matches jaxpi's (losses to 2e-7, raw grad-norm weights to 6 digits, every parameter moving the same way, within 3.8e-6 on steps up to 1e-3; `one_step.py --config no_fourier_feature` and `one_step_ours.py <npz> jaxpi no_fourier`). Over the first 101 updates on the paper's batches (both on CPU, logging every step), our L_ic stays within 1e-5 of jaxpi's, though it swings by ±50% from step to step (1.62, 0.29, 0.07, 0.17, …). jaxpi's own CPU and GPU runs differ by 3e-6 at step 100. But the row amplifies round-off fast: our own CPU and GPU runs differ in L_ic by 4e-5 at step 100, 4e-3 at 200 and 28% at 300. In `default` they still agree to 7e-6 at 300. Our run from the same network on our own batches escaped too (from 52k, ending at 8.10e-2). So 3 of our 8 runs of this row escape, to 0.05–0.08, and 5 stay at 0.41–0.51, like the paper's single run.
  - **jaxpi escapes too, from the paper's exact inputs.** jaxpi's `no_fourier_feature` from seed 42, on its own key-1234 batches (`~/jaxpi-runs/chain6.sh`), matches the paper's run at 1k steps (rel. L2 0.636 against 0.637), then leaves the wrong solution from 39k (below 0.3 from 47.3k, 0.1 from 60.7k) and ends at 4.74e-2. So the paper's run stayed stuck by chance, and our replay went the way jaxpi's run did.
  - From its seeds 0–5, jaxpi escapes in 3 of 6 (seeds 2, 4, 5, ending at 3.3–4.3e-2, below 0.3 from 39–68k) and stays stuck in 3 (4.15e-1, 4.81e-1, 5.08e-1). With seed 42, 4 of 7 escape, against 3 of our first 8 (Fisher p = 0.62), and 8 of our 14 with the six runs below. Over jaxpi's seven runs the median is 4.74e-2, an escaped value; over our fourteen, 7.7e-2.
  - **Escaped, ours and jaxpi's end at the same level.** After the first runs, jaxpi's escaped runs looked lower (3.3–4.7e-2 against our 5.3–8.1e-2, Mann–Whitney p = 0.057, 4 runs against 3). So we ran our `no_fourier` from jaxpi's `no_fourier_feature` seed 0–5 networks, on jaxpi's batches with jaxpi's IC grid (`runs/allen_cahn/nf_batches/seedN/`, `runs\nf_batches.ps1`):

    | jaxpi's network | jaxpi | Ours, same network and batches |
    |---|--:|--:|
    | seed 0 | 5.08e-1 | 5.79e-2 |
    | seed 1 | 4.15e-1 | 4.66e-2 |
    | seed 2 | 4.32e-2 | 5.66e-2 |
    | seed 3 | 4.81e-1 | 2.93e-2 |
    | seed 4 | 3.66e-2 | 2.92e-2 |
    | seed 5 | 3.27e-2 | 2.35e-1 |

    - The outcomes do not pair, as expected from the chaos above. Ours escaped in 5 of 6 and jaxpi in 3 of 6, and both escaped only from seeds 2 and 4. Our seed 5 left the wrong solution late (below 0.3 from 121k, when the learning rate is about 1e-5) and stalled at 0.235; jaxpi's seed 1 did the same at 0.415.
    - Escaped, ours end at 2.9–5.8e-2 (median 4.7e-2), against jaxpi's 3.3–4.7e-2 (median 4.0e-2), Mann–Whitney p = 0.90. Over all escaped runs (8 of ours, 4 of jaxpi's), p = 0.28. The earlier hint is gone.
    - Over all runs of this row, ours escape in 8 of 14 and jaxpi in 4 of 7 (Fisher p = 1.0). On jaxpi's batches, 10 of 14 runs from both trainers escape, against 2 of 7 of ours on our own batches (p = 0.16, and three of those seven are D21 pairs sharing a network and batches).
  - The error of the four replays that reach the right solution sits at x = 0 (0.066–0.102 × the error at (t = 1, x = 0)), and so does the escaped `no_fourier` replay's (0.116×). `plain` is wrong everywhere.
  - **Reading:** given the paper's network and batch sequence, four rows end within 0.89–1.01× of the paper. `no_causal` fell back to the plateau once and ends 3.3× high, and `no_fourier` took the other of its two outcomes. Both are run-to-run events, not steady drifts. jaxpi falls back too (1 of 10 runs of `no_causal` from this network, and from seed 5's network on `default`), and on `no_fourier` jaxpi's own run from the same inputs escapes as ours did. So our trainer reproduces the paper's Table 2 runs, except where a single run's outcome is a coin toss.
- **Running** (queues in `runs\`, the first started 22:48 on 2026-10-31):
  - done: the 30k-step runs above and the `default` and `no_causal` replays. The queue was restarted at 01:32 (PST) on 2026-11-01 as `runs\from_paper_init.ps1` alone, to put the fall-back test ahead of the IC-grid runs. The `no_rwf` replay resumed from its 5k checkpoint (replays resume exactly, since the batch index follows the step);
  - all six replays finished by 05:01. The first round of the fall-back test (jaxpi keys 1–3 via `~/jaxpi-runs/chain4.sh`, ours batch seeds 43–45) finished by 05:52. The second round (`runs\fallback_more.ps1`: jaxpi keys 4–9 via `~/jaxpi-runs/chain5.sh`, ours batch seeds 46–51) finished by 07:36. The queue before it was stopped right after the IC-grid seed 0 run saved step 70k;
  - the IC-grid test's seeds 0–2 finished at 09:57, and seeds 3–5 (`runs\icgrid_more.ps1`) at 12:38;
  - `runs\paper_batches.ps1` (12:45–17:36): our `default` from jaxpi's seed 0–5 networks, on jaxpi's batch sequence with jaxpi's IC grid, to 200k steps. Each run continued the 30k-step replay `runs/allen_cahn/batch_test/replay_seedN/`, copied to `runs/allen_cahn/paper_batches/seedN/` first. About 58 min each;
  - `~/jaxpi-runs/chain6.sh` (17:39 on 2026-11-01 to 00:37 on 2026-11-02): jaxpi's `no_fourier_feature` from seed 42 and seeds 0–5, then `no_grad_norm` and `plain` from seed 42, all 200k steps on its own sampler key. jaxpi's `no_fourier_feature` runs at about 100 steps/s. Logs and final networks are copied to the scratchpad's `jaxpi_logs/`; `chain6_compare.py` reads them;
  - `runs\nf_batches.ps1` (00:39–05:24 on 2026-11-02): our `no_fourier` from jaxpi's `no_fourier_feature` seed 0–5 networks (exported by `export_init.py --config no_fourier_feature`, `init_seed{0..5}_legacy_no_fourier_feature.npz`), on jaxpi's batch sequence with jaxpi's IC grid, 200k steps, into `runs/allen_cahn/nf_batches/seedN/`. `nf_batches_compare.py` reads them;
  - `~/jaxpi-runs/chain7.sh` (05:25–10:42 on 2026-11-02): jaxpi's `default` from seeds 0–5, 200k steps, each on its own sampler key (100 + seed), into `~/jaxpi-runs/default_seedN_highest_legacy_key10N.jsonl`. The logs and final networks are copied to the scratchpad's `jaxpi_logs/`; `keys_compare.py` reads them;
  - now (`runs\own_batches2.ps1`, started 10:48 on 2026-11-02): for each of seeds 0–5, ours from jaxpi's seed-s network on generator seed 101 + s with jaxpi's IC grid (into `runs/allen_cahn/own_batches2/seedN/`), then jaxpi from seed s on sampler key 200 + s (`~/jaxpi-runs/chain8.sh s`, into `default_seedN_highest_legacy_key20N.jsonl`). About 1 h 50 min per seed, until about 22:00. The queue calls WSL itself, so do not start another jaxpi chain alongside it.
  - The scripts `from_jaxpi_init.py` (options `--row`, `--ic-grid`, `--batches`, `--steps`, `--eval-every`), `batch_test_compare.py` and `escape_steps.py`, the exported networks (`init_seed*_legacy*.npz`, also in `~/jaxpi-runs`) and jaxpi's batch sequence (`jaxpi_batches_key1234*.npy`, from `~/jaxpi-runs/export_batches.py`) are in the session scratchpad, outside the repo.
  - The jaxpi chains take about 50 min per run. They skip finished runs, and an interrupted run starts over.
  - Launch `.ps1` queues with a relative `-File` path. `Start-Process` does not quote its arguments, so a full path through "Python Code" breaks at the space and the queue exits at once.
- **Setup:**
  - jaxpi `main` (`f944a0c`) is cloned at `~/jaxpi` in WSL2 Ubuntu, with a virtualenv at `~/jaxpi-venv`. It has jax 0.6.2 (the newest for Python 3.10), CPU-only torch (jaxpi's sampler imports it) and `tabulate`.
  - Our wrapper `~/jaxpi-runs/run_ac.py` replaces W&B with a JSON-lines log (`~/jaxpi-runs/*.jsonl`). Its options: `--seed`, `--precision highest|default`, `--rng legacy` (the old random streams that reproduce the paper's initialization), `--config` (a Table 2 row) and `--sampler-key` (a residual-sampler key other than jaxpi's fixed 1234).
  - Launch from Windows with `Start-Process wsl.exe -ArgumentList "-d","Ubuntu","--exec","bash","/home/maan/jaxpi-runs/chain.sh" -WindowStyle Minimized`.
- **License** (read 2026-10-30): jaxpi may be used, copied and modified for non-profit research only, and not distributed to third parties without Penn's approval. Publications should acknowledge it. Keep its code, and the wrapper, out of this repository, and never imitate its code here.

## Next steps
- **The Table 2 gap is explained.** Given the paper's network and batches, our trainer lands on the paper's values, except in `no_causal` (a fall-back, 1 in 10 for jaxpi too) and `no_fourier` (an escape, which jaxpi's own run on the same inputs makes too; 4 of 7 jaxpi runs and 8 of 14 of ours escape, to the same levels). From our own networks and batches, the paper's favorable seed 42 accounts for 3.6× on `default`, by jaxpi's 12 runs from seeds 0–5. Given the same network and batches, our trainer ends where jaxpi's does from seeds 0–5 and 42 (`paper_batches/`). The rest, about 1.3–1.4×, comes with our own batches: from the same networks, runs on them end higher than runs on jaxpi's batches, from either trainer. That is either chance or a difference in our batch stream not yet found (next bullet). `docs/results.md` is the write-up.
- **Read `runs\own_batches2.ps1`'s runs** (ours in `runs/allen_cahn/own_batches2/seedN/`; jaxpi's `~/jaxpi-runs/default_seedN_highest_legacy_key20N.jsonl`, to be copied to the scratchpad's `jaxpi_logs/`). Add them to `batch_stats.py`.
  - If our runs on fresh streams again end about 1.3× above jaxpi's runs on other keys, and above ours on key 1234, something in our batch stream matters. Look beyond the checks in `sampler_check.py`: feed a jaxpi stream other than key 1234 to our trainer, or one of our streams to jaxpi through the wrapper. Both trainers follow each other given the same batches, so this would show which side the difference is on.
  - If they end like jaxpi's, the first six were chance. Then the 5.0× is the seed (3.6×) and the spread of a five-seed median (our own-seed median is 1.4× jaxpi's).
- **Same-network pairs differ by up to 3.4×** from batches and round-off alone, so read single runs with that in mind.
- **Report Table 2 as distributions.** The paper's values are single seed-42 runs, and for `default` that seed is about 3× better than jaxpi's own typical seed. A fair comparison sets our seed spread against jaxpi's, not against one number.
- **Comparing with the paper's runs:** the W&B histories can be read without an account through `https://api.wandb.ai/graphql` (project `jaxpi/allen_cahn`; run ids in `docs/decisions.md`, JW). They log the error, both losses, both weights and the minimum causal weight every 100 steps. Prefer them over values read off EG Fig. 4.
  - **jaxpi's (and the paper's) log at step s comes after s + 1 updates; ours comes after s.** On the 100-step grid this hardly matters after the first few thousand steps. Early on, compare our step s + 1 with jaxpi's step s, or the losses can look percents apart when they agree to 1e-5. Also, rel. L2 differs by a constant 0.2% between the two, because each scores against its own reference grid.
- **When a run looks off,** check `training.png`:
  - the minimum causal weight should reach 1 by about 30k steps, as in the paper's runs;
  - after the first update, λ_ic is 49 in the paper's run and 56–81 in ours on CUDA (D13).
- **Settings taken from jaxpi** (D2–D13, D20, D21) are matched to its code, and JW confirms D2, D13 and D20. Change them only with evidence, and record any change in `docs/decisions.md`.

## What exists
- **`piratenets/`:**
  - the Allen–Cahn residual, with derivatives via `torch.func` (an `autograd` path is also available);
  - exact periodic embedding, trainable random Fourier features, random weight factorization (RWF) layers, and a plain MLP;
  - causal weighting and grad-norm loss weights;
  - a trainer with CSV logging, and checkpoints whose resume is bit-exact on CPU and on CUDA (both are tested).
- **`configs/allen_cahn/ablation/`:** one YAML per Table 2 row. Each differs from `default.yaml` only in its switch, and a test checks that.
- **`scripts/`:** `train.py`, `run_ablation.py`, `benchmark_step.py`, `plot_ablation.py`, `plot_run.py`, `pack_results.py`, `error_location.py`.
- **`docs/decisions.md`:** every detail the papers leave open (D1–D21), with its source. Read it before changing any setting. `docs/spec.md` describes what is implemented. `docs/results.md` is the short public write-up of the Table 2 reproduction; keep it in step with this file.
- **`notebooks/colab_ablation.ipynb`:** a Colab fallback, no longer needed.
- **Not implemented:** Modified MLP, PirateNet with physics-informed init, NTK weighting, SOAP.

## Settings taken from jaxpi rather than a plain reading of the paper
- 200k steps.
- Fourier features: scale σ = 1, 256 outputs, B trained.
- RWF μ = 0.5.
- Truncated-normal Glorot initialization.
- Causal chunks formed by sorting the sampled times into 32 equal-count groups.
- IC loss on the fixed 512-point reference x-grid.
- Grad-norm weights updated *after* the optimizer step, using the *mean* of the norms.
- RWF trains the scale exp(s) itself, not s.

## Working conventions
- **jaxpi:** values and design decisions may come from jaxpi, but never copy its code. Where the paper and jaxpi disagree, default to jaxpi, implemented independently, and keep the paper's version as a config option. Record each such choice in `docs/decisions.md`.
- **Configs:** every component is a config switch, and each experiment is a YAML file.
- **Plots:** they read the CSV logs and never re-run training.
- **Compute:** this laptop's GPU. Launch long runs as a detached process (`windows_run_guide.md` step 7). They take about 40–50 min per 200k-step row.
- **Python:** call the venv interpreter explicitly (`.\.venv\Scripts\python.exe`); each shell command starts a fresh shell.
- **Papers:** `papers/` is gitignored. Download them from arXiv if needed: 2402.00326 (PirateNets) and 2308.08468 (Expert's Guide).
