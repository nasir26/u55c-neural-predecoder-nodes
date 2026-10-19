"""C simulation of the HLS kernel (hls/predecoder.cpp, compiled with g++ and the Vitis HLS headers) bit exact against
the integer model, on syndrome windows of NVIDIA's inference datapipe.

Author: Nasir Ali, C-DAC Noida.
Checkpoint: QMODEL_CHECKPOINT (default: NVIDIA's model 1 under $NPD_RUNS/models/nvidia if present, else the random
stand-in). Skipped without the Vitis HLS headers.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT / "harness"), str(ROOT / "model"), str(ROOT / "host")]
import nvidia_bridge as nb  # noqa: E402
from pack import CFG, pack_model, pack_windows, unpack_bits  # noqa: E402
from qmodel import PRE_BITS, ACC_SHIFT, calibrate  # noqa: E402

HLS_INC = Path(os.environ.get("XILINX_HLS", "/tools/Xilinx/Vitis_HLS/2023.2")) / "include"
MODELS = nb.RUNS / "models"
NVIDIA = MODELS / "nvidia" / "ising_decoder_surface_code_1_fast_r9_v1.0.77_fp16.safetensors"
CKPT = os.environ.get("QMODEL_CHECKPOINT", str(NVIDIA if NVIDIA.exists() else MODELS / "random_model1_fp16.safetensors"))


def test_config_agrees_with_model():
    assert CFG["PRE_BITS"] == PRE_BITS and CFG["ACC_SHIFT"] == ACC_SHIFT and CFG["C"] == 128


@pytest.fixture(scope="module")
def tb(tmp_path_factory):
    if not (HLS_INC / "ap_int.h").exists():
        pytest.skip("Vitis HLS headers not installed")
    exe = tmp_path_factory.mktemp("npd") / "tb"
    subprocess.run(["g++", "-std=c++17", "-O2", "-Wno-unknown-pragmas", f"-I{HLS_INC}", f"-I{ROOT / 'hls'}",
                    str(ROOT / "hls" / "predecoder.cpp"), str(ROOT / "hls" / "tb_predecoder.cpp"), "-o", str(exe)],
                   check=True, capture_output=True)
    return exe


@pytest.mark.parametrize("basis,d,t", [("X", 9, 9), ("Z", 9, 9), ("X", 11, 9)])
def test_kernel_bit_exact(tb, tmp_path, basis, d, t):
    cfg = nb.make_cfg(d, t)
    model, _ = nb.load_reference_model(CKPT)
    model = model.float()
    X = torch.stack([b["trainX"] for _, b in zip(range(67), nb.sample_windows(cfg, basis, 67))])
    qm = calibrate(model, X[:64])
    xi = np.rint(X[64:].numpy() * 2).astype(np.int64)
    want, _ = qm.forward_int_fast(xi)
    wmem, params = pack_model(qm)
    wmem.tofile(tmp_path / "w.bin"); params.tofile(tmp_path / "p.bin"); pack_windows(xi).tofile(tmp_path / "x.bin")
    subprocess.run([str(tb), str(tmp_path / "w.bin"), str(tmp_path / "p.bin"), str(tmp_path / "x.bin"), str(len(xi)),
                    str(d), str(t), str(tmp_path / "b.bin")], check=True)
    got = unpack_bits(np.fromfile(tmp_path / "b.bin", dtype=np.uint8).reshape(len(xi), -1), t, d)
    assert np.array_equal(got, want)


ACCURATE = MODELS / "nvidia" / "ising_decoder_surface_code_1_accurate_r13_v1.0.86_fp16.safetensors"


@pytest.mark.skipif(not ACCURATE.exists(), reason="NVIDIA's Accurate checkpoint (model 4, gated) not downloaded")
def test_kernel_bit_exact_model4(tb, tmp_path):
    """The same build with L = 6 layers against the integer model of NVIDIA's model 4 (Accurate), d = 13, 13 rounds."""
    d = t = 13
    cfg = nb.make_cfg(d, t, model_id=4)
    model, _ = nb.load_reference_model(str(ACCURATE))
    model = model.float()
    X = torch.stack([b["trainX"] for _, b in zip(range(35), nb.sample_windows(cfg, "X", 35))])
    qm = calibrate(model, X[:32])
    assert len(qm.layers) == 6
    xi = np.rint(X[32:].numpy() * 2).astype(np.int64)
    want, _ = qm.forward_int_fast(xi)
    wmem, params = pack_model(qm)
    wmem.tofile(tmp_path / "w.bin"); params.tofile(tmp_path / "p.bin"); pack_windows(xi).tofile(tmp_path / "x.bin")
    subprocess.run([str(tb), str(tmp_path / "w.bin"), str(tmp_path / "p.bin"), str(tmp_path / "x.bin"), str(len(xi)),
                    str(d), str(t), str(tmp_path / "b.bin")], check=True)
    got = unpack_bits(np.fromfile(tmp_path / "b.bin", dtype=np.uint8).reshape(len(xi), -1), t, d)
    assert np.array_equal(got, want)
