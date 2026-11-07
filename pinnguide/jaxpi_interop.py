"""Load inputs exported from jaxpi, the authors' JAX code, to rerun its experiments in this trainer.

Two kinds of file, both written from a jaxpi checkout by the scripts in scripts/jaxpi/ (which need JAX and
jaxpi; this module needs neither):

- an initial network (export_init.py): jaxpi's parameters as flat arrays named like
  "params/Dense_0/kernel", a factorized kernel split into ".../kernel/g" and ".../kernel/v", plus jaxpi's
  prediction at a few random points (check_t, check_x, check_u) to verify the mapping;
- a batch sequence (export_batches.py): the residual points jaxpi's sampler draws, one batch per step, as a
  float32 array of shape (steps, batch_size, 2) with columns t, x.

scripts/replay_jaxpi.py puts them together.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

# jaxpi fits the initial condition on the x-grid of its reference data, linspace(-1, 1, 512): x = -1 and
# x = 1 are both included (the same point, by periodicity) and x = 0 is not. This code's own grid is 512
# periodic points on [-1, 1).
JAXPI_IC_X = np.linspace(-1, 1, 512, dtype=np.float32)


def jaxpi_param_map(model: nn.Module, arrays) -> dict[str, np.ndarray]:
    """Map exported jaxpi arrays to this model's parameter names (Dense kernels transposed to (out, in)).

    `arrays` is anything indexable by name with a `files` list or `keys()`, such as an np.load result.
    Raises ValueError if the two do not hold the same parameters with the same shapes.
    """
    names = set(arrays.files) if hasattr(arrays, "files") else set(arrays.keys())
    params = dict(model.named_parameters())
    pairs = {}
    if "params/FourierEmbs_0/kernel" in names:
        pairs["fourier.B"] = arrays["params/FourierEmbs_0/kernel"].T
    layers = [f"hidden.{j}" for j in range(len(model.hidden))] + ["out"]
    for i, name in enumerate(layers):
        key = f"params/Dense_{i}"
        if f"{key}/kernel/g" in names:  # factorized kernel (random weight factorization)
            pairs[f"{name}.g"] = arrays[f"{key}/kernel/g"]
            pairs[f"{name}.v"] = arrays[f"{key}/kernel/v"].T
        elif f"{key}/kernel" in names:
            pairs[f"{name}.weight"] = arrays[f"{key}/kernel"].T
        if f"{key}/bias" in names:
            pairs[f"{name}.bias"] = arrays[f"{key}/bias"]
    if set(pairs) != set(params):
        raise ValueError(
            "the exported network does not match this model (was it exported from the matching jaxpi config?): "
            f"only in the export {sorted(set(pairs) - set(params))}, only in the model {sorted(set(params) - set(pairs))}"
        )
    for k, v in pairs.items():
        if tuple(params[k].shape) != v.shape:
            raise ValueError(f"{k}: the model has shape {tuple(params[k].shape)}, the export {v.shape}")
    return pairs


def load_jaxpi_params(model: nn.Module, arrays) -> None:
    """Copy an exported jaxpi network into `model` in place."""
    pairs = jaxpi_param_map(model, arrays)
    params = dict(model.named_parameters())
    with torch.no_grad():
        for k, v in pairs.items():
            params[k].copy_(torch.as_tensor(np.ascontiguousarray(v)))


class BatchFile:
    """jaxpi's residual batches, read from an exported .npy file one step at a time.

    Step k returns batch k as (t, x), each of shape (num_chunks, batch_size // num_chunks). With causal
    weighting the times are sorted and x keeps its order, as jaxpi pairs them; without it neither is
    reordered.
    """

    def __init__(self, path, num_chunks: int, sort_t: bool, device):
        self.bank = np.load(path, mmap_mode="r")
        if self.bank.ndim != 3 or self.bank.shape[2] != 2:
            raise ValueError(f"{path}: expected an array of shape (steps, batch_size, 2), got {self.bank.shape}")
        if self.bank.shape[1] % num_chunks:
            raise ValueError(f"batch size {self.bank.shape[1]} must be divisible by num_chunks {num_chunks}")
        self.num_chunks = num_chunks
        self.sort_t = sort_t
        self.device = device

    def __len__(self) -> int:
        return self.bank.shape[0]

    @property
    def batch_size(self) -> int:
        return self.bank.shape[1]

    def __call__(self, step: int):
        if step >= len(self):
            raise IndexError(f"the batch file holds {len(self)} steps, step {step} was requested")
        b = torch.as_tensor(np.array(self.bank[step]), device=self.device)
        t, x = b[:, 0], b[:, 1]
        if self.sort_t:
            t = torch.sort(t).values
        return t.reshape(self.num_chunks, -1), x.reshape(self.num_chunks, -1)
