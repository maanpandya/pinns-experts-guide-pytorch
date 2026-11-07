"""Save the residual batches jaxpi's Allen-Cahn training draws, step by step, for scripts/replay_jaxpi.py.

Runs in a Python environment with JAX and jaxpi (https://github.com/PredictiveIntelligenceLab/jaxpi), not
in this package's environment. jaxpi is used unmodified and is not part of this repository.

    JAX_PLATFORMS=cpu python export_batches.py --jaxpi ~/jaxpi --steps 200000 --out jaxpi_batches.npy

Draws from jaxpi's own UniformSampler on the domain its train.py builds, with its fixed key (1234) and the
random streams JAX used before 0.5, as the paper's runs did. Every Table 2 row trains on this one sequence.
The output is one float32 array of shape (steps, 4096, 2), columns t and x, unsorted: 6.5 GB for 200k steps,
written to disk as it is drawn. Tested with jax 0.6.2 and jaxpi at commit f944a0c.
"""
import argparse
import os
import sys

import numpy as np

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--jaxpi", default="~/jaxpi", help="path to a jaxpi checkout")
p.add_argument("--steps", type=int, default=200_000)
p.add_argument("--batch-size", type=int, default=4096)
p.add_argument("--sampler-key", type=int, default=1234)
p.add_argument("--out", required=True)
a = p.parse_args()

import jax  # noqa: E402

jax.config.update("jax_threefry_partitionable", False)
import jax.numpy as jnp  # noqa: E402

example = os.path.join(os.path.expanduser(a.jaxpi), "examples", "allen_cahn")
out = os.path.abspath(os.path.expanduser(a.out))
os.chdir(example)  # jaxpi's example loads its data by a relative path
sys.path.insert(0, example)
from jaxpi.samplers import UniformSampler  # noqa: E402
from utils import get_dataset  # noqa: E402  (jaxpi's examples/allen_cahn/utils.py)

u_ref, t_star, x_star = get_dataset()
dom = jnp.array([[t_star[0], t_star[-1]], [x_star[0], x_star[-1]]])
sampler = UniformSampler(dom, a.batch_size, rng_key=jax.random.PRNGKey(a.sampler_key))
batches = np.lib.format.open_memmap(out, mode="w+", dtype=np.float32, shape=(a.steps, a.batch_size, 2))
for k in range(a.steps):
    batches[k] = np.asarray(sampler[k])[0]  # drop the device axis
batches.flush()
print("domain", np.asarray(dom).tolist(), "first batch", batches[0, :3].tolist(), "saved", out)
