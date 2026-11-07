"""Save the initial network of jaxpi's Allen-Cahn example as plain arrays, for scripts/replay_jaxpi.py.

Runs in a Python environment with JAX and jaxpi (https://github.com/PredictiveIntelligenceLab/jaxpi), not
in this package's environment. jaxpi is used unmodified and is not part of this repository.

    JAX_PLATFORMS=cpu python export_init.py --jaxpi ~/jaxpi --config default --seed 42 --out init_default.npz

--config names a file in jaxpi's examples/allen_cahn/configs/ (default, no_causal, no_rwf,
no_fourier_feature, plain). The paper's runs used seed 42 with the random streams JAX used before 0.5
(--rng legacy, the default); with the newer streams the same seed builds a different network.
Tested with jax 0.6.2 and jaxpi at commit f944a0c.

Also saves jaxpi's prediction at 500 random (t, x) points, which replay_jaxpi.py compares with its own.
"""
import argparse
import importlib.util
import os
import sys

import numpy as np

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--jaxpi", default="~/jaxpi", help="path to a jaxpi checkout")
p.add_argument("--config", default="default")
p.add_argument("--seed", type=int, default=42)
p.add_argument("--rng", choices=["legacy", "current"], default="legacy")
p.add_argument("--out", required=True)
a = p.parse_args()

import jax  # noqa: E402

jax.config.update("jax_default_matmul_precision", "highest")
if a.rng == "legacy":
    jax.config.update("jax_threefry_partitionable", False)
from flax.traverse_util import flatten_dict  # noqa: E402

example = os.path.join(os.path.expanduser(a.jaxpi), "examples", "allen_cahn")
out = os.path.abspath(os.path.expanduser(a.out))
os.chdir(example)  # jaxpi's example loads its data by a relative path
sys.path.insert(0, example)
spec = importlib.util.spec_from_file_location("ac_config", os.path.join(example, "configs", f"{a.config}.py"))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
config = mod.get_config()
config.seed = a.seed

import models  # noqa: E402  (jaxpi's examples/allen_cahn/models.py)
from utils import get_dataset  # noqa: E402  (jaxpi's examples/allen_cahn/utils.py)

u_ref, t_star, x_star = get_dataset()
model = models.AllenCahn(config, u_ref[0, :], t_star, x_star)
params = jax.device_get(jax.tree_util.tree_map(lambda x: x[0], model.state.params))  # drop the device axis
arrays = {}
for k, v in flatten_dict(params).items():
    name = "/".join(k)
    if isinstance(v, (tuple, list)):  # a factorized kernel is stored as the pair (g, v)
        arrays[name + "/g"], arrays[name + "/v"] = (np.asarray(e) for e in v)
    else:
        arrays[name] = np.asarray(v)

rng = np.random.default_rng(0)
t = rng.uniform(0, 1, 500).astype(np.float32)
x = rng.uniform(-1, 1, 500).astype(np.float32)
u = np.asarray(jax.vmap(model.u_net, (None, 0, 0))(params, t, x))
np.savez(out, check_t=t, check_x=x, check_u=u, **arrays)
for k, v in arrays.items():
    print(k, v.shape)
print("saved", out)
