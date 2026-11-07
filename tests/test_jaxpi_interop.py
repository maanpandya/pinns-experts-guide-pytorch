import numpy as np
import pytest
import torch

from pinnguide.jaxpi_interop import JAXPI_IC_X, BatchFile, jaxpi_param_map, load_jaxpi_params
from pinnguide.models import build_model
from pinnguide.problems import allen_cahn as ac


def _jaxpi_arrays(model):
    """Write a model's parameters the way scripts/jaxpi/export_init.py writes jaxpi's (kernels as (in, out))."""
    p = {k: v.detach().numpy() for k, v in model.named_parameters()}
    arrays = {}
    if "fourier.B" in p:
        arrays["params/FourierEmbs_0/kernel"] = p["fourier.B"].T
    for i, name in enumerate([f"hidden.{j}" for j in range(len(model.hidden))] + ["out"]):
        if f"{name}.g" in p:
            arrays[f"params/Dense_{i}/kernel/g"] = p[f"{name}.g"]
            arrays[f"params/Dense_{i}/kernel/v"] = p[f"{name}.v"].T
        else:
            arrays[f"params/Dense_{i}/kernel"] = p[f"{name}.weight"].T
        arrays[f"params/Dense_{i}/bias"] = p[f"{name}.bias"]
    return arrays


@pytest.mark.parametrize("fourier,rwf", [(True, True), (False, False), (True, False), (False, True)])
def test_loads_a_jaxpi_network_exactly(tiny_cfg, fourier, rwf):
    tiny_cfg.model.fourier.enabled = fourier
    tiny_cfg.model.rwf.enabled = rwf
    torch.manual_seed(0)
    source = build_model(tiny_cfg.model, x_period=ac.X_PERIOD)
    torch.manual_seed(1)
    target = build_model(tiny_cfg.model, x_period=ac.X_PERIOD)
    t, x = torch.rand(50), 2 * torch.rand(50) - 1
    assert not torch.allclose(source(t, x), target(t, x))

    load_jaxpi_params(target, _jaxpi_arrays(source))
    for (k, a), (_, b) in zip(source.named_parameters(), target.named_parameters()):
        assert torch.equal(a, b), k
    assert torch.equal(source(t, x), target(t, x))


def test_rejects_a_network_from_another_config(tiny_cfg):
    with_rwf = build_model(tiny_cfg.model, x_period=ac.X_PERIOD)
    tiny_cfg.model.rwf.enabled = False
    without_rwf = build_model(tiny_cfg.model, x_period=ac.X_PERIOD)
    with pytest.raises(ValueError, match="does not match"):
        jaxpi_param_map(without_rwf, _jaxpi_arrays(with_rwf))
    tiny_cfg.model.hidden_dim += 1
    wider = build_model(tiny_cfg.model, x_period=ac.X_PERIOD)
    with pytest.raises(ValueError, match="shape"):
        jaxpi_param_map(wider, _jaxpi_arrays(without_rwf))


def test_batch_file_serves_batch_k(tmp_path):
    rng = np.random.default_rng(0)
    bank = np.stack([rng.uniform(0, 1, (3, 8)), rng.uniform(-1, 1, (3, 8))], axis=-1).astype(np.float32)
    np.save(tmp_path / "b.npy", bank)

    causal = BatchFile(tmp_path / "b.npy", num_chunks=2, sort_t=True, device="cpu")
    assert len(causal) == 3 and causal.batch_size == 8
    t, x = causal(1)
    # Under causal weighting the times are sorted and x keeps its order, as jaxpi pairs them.
    assert torch.equal(t, torch.as_tensor(np.sort(bank[1, :, 0])).reshape(2, 4))
    assert torch.equal(x, torch.as_tensor(bank[1, :, 1]).reshape(2, 4))

    plain = BatchFile(tmp_path / "b.npy", num_chunks=2, sort_t=False, device="cpu")
    assert torch.equal(plain(2)[0], torch.as_tensor(bank[2, :, 0]).reshape(2, 4))
    with pytest.raises(IndexError):
        plain(3)
    with pytest.raises(ValueError, match="divisible"):
        BatchFile(tmp_path / "b.npy", num_chunks=3, sort_t=True, device="cpu")


def test_jaxpi_ic_grid_holds_both_endpoints():
    assert JAXPI_IC_X.shape == (512,) and JAXPI_IC_X[0] == -1 and JAXPI_IC_X[-1] == 1
    assert not np.any(JAXPI_IC_X == 0)
