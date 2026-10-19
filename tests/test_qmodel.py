"""The integer model: fast path bit identical to the reference path, requantization and table invariants.

Author: Nasir Ali, C-DAC Noida.

Runs on any model 1 checkpoint in NVIDIA's SafeTensors format (QMODEL_CHECKPOINT, default: the random stand-in under
$NPD_RUNS/models), on syndrome windows from NVIDIA's inference datapipe at d = 9 and 9 rounds.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / "harness"), str(HERE.parent / "model")]
import nvidia_bridge as nb  # noqa: E402
from qmodel import PRE_BITS, calibrate  # noqa: E402

CKPT = os.environ.get("QMODEL_CHECKPOINT", str(nb.RUNS / "models" / "random_model1_fp16.safetensors"))


@pytest.fixture(scope="module")
def setup():
    cfg = nb.make_cfg(9, 9)
    model, _ = nb.load_reference_model(CKPT)
    model = model.float()
    X = torch.stack([b["trainX"] for _, b in zip(range(160), nb.sample_windows(cfg, "X", 160))])
    return model, calibrate(model, X[:128]), X[128:]


def test_inputs_are_small_integers(setup):
    _, _, X = setup
    xi = X.numpy() * 2
    assert np.array_equal(xi, np.rint(xi)) and xi.min() >= 0 and xi.max() <= 2


def test_fast_path_bit_identical(setup):
    _, qm, X = setup
    xi = np.rint(X[:16].numpy() * 2).astype(np.int64)
    b1, a1 = qm.forward_int(xi)
    b2, a2 = qm.forward_int_fast(xi)
    assert np.array_equal(a1, a2) and np.array_equal(b1, b2)


def test_quantized_ranges(setup):
    _, qm, _ = setup
    for L in qm.layers:
        assert L.w.dtype == np.int8 and np.abs(L.w.astype(int)).max() <= 127
    for L in qm.layers[:-1]:
        assert L.lut.shape == (1 << PRE_BITS,) and L.lut.dtype == np.int8
        assert np.all(np.diff(L.lut[(1 << (PRE_BITS - 1)):].astype(int)) >= 0)    # GELU is monotone for x >= 0
        assert L.m.max() < (1 << 31)                                                # fits a 32 bit multiplier
    assert qm.layers[-1].thr is not None
