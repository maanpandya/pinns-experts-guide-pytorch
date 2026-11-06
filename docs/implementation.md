# Implementation details

What this code implements, and the value and source of every setting the papers leave open. "EG" is the
Expert's Guide (arXiv:2308.08468) and "PN" is PirateNets (arXiv:2402.00326). Where neither paper fixes a
detail, the authors' JAX code ([jaxpi](https://github.com/PredictiveIntelligenceLab/jaxpi)) and its example
configs for this ablation decide it.

## Problem

    u_t − 1e-4 u_xx + 5u³ − 5u = 0,   t ∈ [0, 1],  x ∈ [−1, 1] periodic,   u(0, x) = x² cos(πx)

The reference solution, `data/allen_cahn_ref.npz`, holds 201 × 512 values on t = 0, 0.005, …, 1 and
x_j = −1 + 2j/512. `data/generate_allen_cahn.py` computes it with ETDRK4 Fourier pseudo-spectral time
stepping at N = 4096, self-converged to about 1e-6. It agrees with the authors' reference data to 1.5e-5
relative L2. The metric is the relative L2 error ‖u_θ − u_ref‖ / ‖u_ref‖ over that grid.

## Model

`piratenets/embeddings.py`, `layers.py`, `models.py`:

1. **Periodic embedding** (EG Sec. 6.3): (t, x) ↦ [t, cos(πx), sin(πx)]. Every x-derivative of the network
   is then exactly periodic, so the loss has no boundary term. All six rows use it, including `plain`.
2. **Random Fourier features** (EG Sec. 4.2): γ(v) = [cos(Bv), sin(Bv)], with B ∈ R^{128×3},
   B_ij ~ N(0, σ²) at initialization. They act on the periodic embedding's output, and B is trained.
3. **MLP**: 4 hidden layers of 256 tanh units and a linear output.
4. **Random weight factorization** (EG Sec. 4.3, Algorithm 2), on every dense layer, the output layer
   included: W = diag(exp(s)) V, with one scale per output neuron. It starts from the Glorot W, with
   s ~ N(μ, σ²) and V = W / exp(s).

## Training

Each step samples 4,096 residual points uniformly in (t, x) and computes:

    L_ic  = mean_i (u_θ(0, x_i) − x_i² cos(π x_i))²        on the 512 x-points of the reference grid
    R     = u_t − 1e-4 u_xx + 5u³ − 5u
    L_r^k = mean of R² over causal chunk k,  k = 1..M        (points sorted by t, M equal-size chunks)
    w_k   = exp(−ε Σ_{j<k} L_r^j)                           (no gradient; w ≡ 1 without causal weighting)
    L_r   = (1/M) Σ_k w_k L_r^k
    L     = λ_ic L_ic + λ_r L_r

**Grad-norm weights** (EG Algorithm 1, eqs. 2.12–2.15): every 1,000 steps the trainer computes
λ̂_k = mean_j ‖∇_θ L_j‖ / ‖∇_θ L_k‖ and sets λ_k ← 0.9 λ_k + 0.1 λ̂_k. λ is a constant for
back-propagation.

**Derivatives:** `training.derivatives: autograd` uses `torch.autograd.grad` with `create_graph=True`.
`func` computes them per point under `torch.func.vmap`: (u_t, u_x) by `grad`, and u_xx by `jvp` of that
gradient. Both give the same values to float32 round-off. With `training.compile: true` the step is
compiled with `torch.compile`, and always uses `func`, because `torch.compile` cannot trace the double
backward of `autograd`. On an RTX 3070 Ti Laptop GPU, before it heats up, a step takes 12.3 ms with
`autograd`, 16.9 ms with `func`, and 7.5 ms compiled (`scripts/benchmark_step.py`). A compiled run of
`default` from seed 0 ends at 3.11e-3 against 2.93e-3 eager, from the same network and batches; the two
agree to 5 digits for the first 2,000 steps and then drift apart from round-off. On GPU, compiling needs Triton: it comes with
PyTorch's CUDA wheels on Linux, and is `pip install triton-windows` on Windows.

## Settings the paper leaves open

| Setting | Value | Source |
|---|---|---|
| Training steps | 200,000 | authors' config |
| Fourier features | σ = 1, 128 frequencies (256 outputs), B trained, after the periodic embedding | authors' config and code |
| RWF | μ = 0.5, σ = 0.1, on every dense layer including the output layer | authors' config (EG gives μ = 0.5 and 1 in different places) and code |
| Initialization | Glorot normal truncated at ±2 std and rescaled to Var(W) = 2/(fan_in + fan_out); zero biases | authors' code (JAX's `glorot_normal`) |
| Learning rate | 1e-3 · 0.9^(step / 2000), decayed continuously; no warmup or weight decay | EG Sec. 7; authors' code |
| Causal chunks | points sorted by t into M = 32 chunks of equal count | authors' config and code |
| Initial-condition points | the 512 x-points of the reference grid, at every step | authors' code |
| Grad-norm update | after the optimizer step, on the same batch, at steps 0, 1000, 2000, …; norms over all trainable parameters, including B; the residual norm is of the causally weighted L_r | authors' code; the logged runs show the same weights |
| `plain` row of Table 2 | printed as 51.74 × 10⁻¹; read as 5.174 × 10⁻¹ | the logged run ends at 0.5174 |

## Where the paper and the authors' code differ

Both versions are config switches. The default follows the authors' code.

| Switch | Default (authors' code) | Alternative (paper) | Effect |
|---|---|---|---|
| `weighting.grad_norm_reduction` | `mean` of the gradient norms | `sum`, as printed in EG eqs. 2.12–2.14 | With two loss terms the mean halves both λ̂, and their ratio, which is what Adam responds to, is unchanged. The two differ only while the moving average still remembers λ = 1, about the first 20k steps. The logged runs used the mean: their λ_r settles at 0.51, which the sum cannot go below 1. |
| `model.rwf.scale_param` | `direct`: trains g = exp(s) | `log`: trains s (EG Algorithm 2) | The same network function with different gradients for the scales. Over 14 paired runs it made no consistent difference to the final error. |

## Numerics and reproducibility

- **Precision:** float32 with full-precision matmuls; TF32 is off, since it would cap the error near 1e-3.
- **Seeds:** `training.seed` seeds the network's initialization. The collocation sampler has its own
  generator, seeded with `training.seed + 1`, on the training device. CPU and CUDA generators draw
  different streams, so a seed gives the same network but different batches on CPU and on GPU.
- **Checkpoints and resume:** every 5,000 steps the trainer writes `<out_dir>/checkpoints/last.pt`
  atomically, holding the model, optimizer, scheduler, loss weights and every RNG state. A resumed run
  continues exactly; the tests check it against an uninterrupted run, bit for bit, on CPU and CUDA.
- **Outputs:** `metrics.csv` (losses, λ, gradient norms, causal weights every 100 steps; rel. L2 every
  1,000), `config.yaml`, and at the end `prediction.npz` (u_θ on the reference grid) and `summary.json`.

## Not implemented

The Modified MLP (EG Sec. 6.4), PirateNets and their physics-informed initialization (PN Sec. 4), NTK
loss weighting, and the settings of EG's best Allen–Cahn run (Table 6).
