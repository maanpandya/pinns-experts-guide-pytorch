# pinns-experts-guide-pytorch

A PyTorch implementation of the physics-informed neural network (PINN) training pipeline from
[*An Expert's Guide to Training Physics-informed Neural Networks*](https://arxiv.org/abs/2308.08468)
(Wang, Sankaran, Wang & Perdikaris, 2023), with a full reproduction of the paper's Allen–Cahn ablation
(Table 2).

<p align="center">
  <img src="docs/table2.png" width="880" alt="Final relative L2 error of each Table 2 row, from the paper's own inputs and from other seeds">
</p>

- **The whole pipeline, one switch per component:** exact periodic embedding, random Fourier features,
  random weight factorization, grad-norm loss balancing and causal weighting.
- **Reproduces Table 2** from the paper's own starting network and batches, and matches the authors' JAX
  code across seeds ([results](docs/results.md)).
- **Fast:** with `torch.compile`, a training step takes 7.5 ms on a laptop GPU, 1.6× faster than eager
  mode.
- **Exact resume:** checkpoints hold every RNG state, so an interrupted run continues bit for bit.

## Results

Relative L2 error after 200k steps:

| Row | Paper | Ours, from the paper's inputs | Ours, median over seeds |
|---|--:|--:|--:|
| `default` (full pipeline) | 5.84e-4 | 5.56e-4 | 2.60e-3 (10 seeds) |
| `no_causal` | 1.59e-3 | 5.26e-3 | 5.16e-3 |
| `no_rwf` | 6.62e-3 | 6.72e-3 | 1.20e-2 |
| `no_grad_norm` | 7.51e-3 | 6.66e-3 | 6.14e-3 |
| `no_fourier` | 4.35e-1 | 7.37e-2 | 5.06e-1 |
| `plain` | 5.17e-1 | 4.85e-1 | 5.08e-1 |

The paper reports one run per row, all from seed 42. Started from that run's network and batch sequence,
four rows land within 0.89–1.01× of the paper. In `no_causal` and `no_fourier` a run can end in one of two
ways, and the authors' code, from the same inputs, also departs from the paper on `no_fourier`. From other
seeds, `default` ends higher than the paper, and so does the authors' code: a median of 2.1e-3 from its
seeds 0–5, against 2.6e-3 here (Mann–Whitney p = 0.34). Seed 42 is a favorable seed.
[`docs/results.md`](docs/results.md) has the details.

## Installation

```bash
git clone https://github.com/maanpandya/pinns-experts-guide-pytorch.git
cd pinns-experts-guide-pytorch
pip install -e .          # or ".[dev]" to run the tests
```

Python 3.10+ and PyTorch 2.2+. A CUDA GPU is recommended. `torch.compile` on GPU needs Triton, which comes
with PyTorch's CUDA wheels on Linux; on Windows, `pip install -e ".[compile]"` adds it (`triton-windows`).

## Quick start

```bash
# Train the full pipeline (Table 2, row 1), compiled
python scripts/train.py --config configs/allen_cahn/ablation/default.yaml --set training.compile=true

# Plot its training curves and the solution against the reference
python scripts/plot_run.py runs/allen_cahn/ablation/ablation_default
```

A run writes `metrics.csv`, `config.yaml`, a checkpoint every 5,000 steps, and at the end
`prediction.npz` and `summary.json`, to `runs/allen_cahn/ablation/<name>/`. Started again, it resumes
from its last checkpoint.

## Reproducing Table 2

```bash
# All six rows, seed 0 (up to about 45 min per row on an RTX 3070 Ti Laptop GPU)
python scripts/run_ablation.py --set training.compile=true

# More seeds, each in its own folder
python scripts/run_ablation.py --set training.compile=true training.seed=1 --out-root runs/allen_cahn/seed1

# Error curves and a comparison table; the Table 2 figure from every seed
python scripts/plot_ablation.py --out-root runs/allen_cahn/ablation
python scripts/plot_table2.py
```

Finished rows are skipped, so the same command picks up where it stopped.

### From the paper's own inputs

The "from the paper's inputs" results start from jaxpi's seed-42 networks and its batch sequence. Export
them with jaxpi (in a JAX environment, with a [jaxpi](https://github.com/PredictiveIntelligenceLab/jaxpi)
checkout), then replay each row in this trainer:

```bash
# In the JAX environment: one network per jaxpi config, and the batch sequence (6.5 GB)
python scripts/jaxpi/export_init.py --jaxpi ~/jaxpi --config default --out init_default.npz
python scripts/jaxpi/export_batches.py --jaxpi ~/jaxpi --out jaxpi_batches.npy

# Here
python scripts/replay_jaxpi.py --row default --init init_default.npz --batches jaxpi_batches.npy
```

`scripts/replay_jaxpi.py --help` lists which jaxpi config each row starts from.

## Configuration

Each experiment is a YAML file in `configs/`, and any value can be overridden from the command line with
`--set key=value`. Unknown keys are an error, so a typo never falls back to a default silently.

| Key | Values | |
|---|---|---|
| `model.fourier.enabled` | `true` / `false` | random Fourier features |
| `model.rwf.enabled` | `true` / `false` | random weight factorization |
| `weighting.scheme` | `grad_norm` / `none` | grad-norm loss balancing |
| `causal.enabled` | `true` / `false` | causal weighting |
| `training.seed` | integer | network initialization; batches use `seed + 1` |
| `training.steps` | integer | 200,000 in the paper |
| `training.compile` | `true` / `false` | compile each training step |
| `weighting.grad_norm_reduction` | `mean` / `sum` | `mean` as in the authors' code, `sum` as printed in the paper |
| `model.rwf.scale_param` | `direct` / `log` | train exp(s) as in the authors' code, or s as in the paper |

[`docs/implementation.md`](docs/implementation.md) lists every setting, its value and its source.

## Performance

One training step of `default` on an RTX 3070 Ti Laptop GPU (`python scripts/benchmark_step.py`):

| Mode | ms/step | 200k steps |
|---|--:|--:|
| `training.derivatives=autograd` (default) | 12.3 | 41 min |
| `training.derivatives=func` | 16.9 | 56 min |
| `training.compile=true` | 7.5 | 25 min |

All three modes give the same values up to float32 round-off. These times are for a cool GPU and count
training steps only. Over a whole run this laptop's GPU heats to its limit (89 °C) and drops its clock from
1.8 to about 0.6 GHz, so a full 200k-step run of `default`, with evaluation and logging, took 42.5 minutes
compiled and 50.8 eager. The paper reports about 16 minutes per run with JAX on an RTX A6000.

## Project layout

```
pinnguide/             model (embeddings, RWF layers, MLP), loss weighting, causal weights, trainer
pinnguide/problems/    Allen–Cahn: PDE residual, sampling, reference data
configs/               one YAML per experiment, one per Table 2 row
scripts/               train, run_ablation, replay_jaxpi, benchmark_step, plot_run, plot_ablation, plot_table2, error_location
scripts/jaxpi/         export jaxpi's initial networks and batches (run with JAX and jaxpi installed)
data/                  reference solution and the script that generates it
docs/                  results.md, implementation.md
tests/                 pytest suite (pytest -q)
```

Not yet implemented: the PirateNet architecture and its physics-informed initialization, the Modified MLP,
and NTK loss weighting.

## Citation

If you use this code, please cite the paper it implements:

```bibtex
@article{wang2023expert,
  title   = {An Expert's Guide to Training Physics-informed Neural Networks},
  author  = {Wang, Sifan and Sankaran, Shyam and Wang, Hanwen and Perdikaris, Paris},
  journal = {arXiv preprint arXiv:2308.08468},
  year    = {2023}
}
```

Settings the paper leaves open follow the authors' JAX implementation,
[jaxpi](https://github.com/PredictiveIntelligenceLab/jaxpi).

## License

MIT, see [LICENSE](LICENSE).
