"""Packs the integer model (model/qmodel.py) and syndrome windows into the memory layouts of hls/predecoder.cpp.

Author: Nasir Ali, C-DAC Noida.

  wmem    uint8 array of W_WORDS * PI bytes: word ((l * TAPS + tap) * CB_IN + cib) * C + o, byte i = weight of output
          channel o from input channel cib * PI + i at tap (dt, dy, dx) = (tap // 9, tap // 3 % 3, tap % 3) of layer l;
          channels that do not exist (layer 1 inputs >= 4, layer 4 outputs >= 4) are zero
  params  int32: bias[L][C], mult[L - 1][C], thr[C_OUT], lut[L - 1][LUT_SIZE] for a model of L = 4 or 6 layers
  x_in    uint8 per site: input channel k (value 0, 1 or 2) in bits 2k, 2k + 1
  bits    uint8 per site: output channel k in bit k
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

CFG_FILE = Path(__file__).resolve().parents[1] / "hls" / "predecoder_config.hpp"


def read_config() -> dict:
    vals: dict[str, int] = {}
    for name, expr in re.findall(r"constexpr int (\w+) = ([^;]+);", CFG_FILE.read_text()):
        vals[name] = int(eval(expr, {}, vals))  # noqa: S307 (constant expressions of this repository's header)
    return vals


CFG = read_config()


def pack_model(qm) -> tuple[np.ndarray, np.ndarray]:
    L, TAPS, C, PI, CB_IN = len(qm.layers), CFG["TAPS"], CFG["C"], CFG["PI"], CFG["CB_IN"]
    assert L in (4, 6), "the kernel is built for 4 (model 1) or 6 (model 4) layers"
    W = np.zeros((L, TAPS, CB_IN, C, PI), dtype=np.int8)
    for l, layer in enumerate(qm.layers):
        w = layer.w                                         # (out, in, 3, 3, 3)
        co, ci = w.shape[:2]
        wt = w.reshape(co, ci, TAPS)                        # tap = dt * 9 + dy * 3 + dx (C order of the kernel)
        for cib in range((ci + PI - 1) // PI):
            n = min(PI, ci - cib * PI)
            W[l, :, cib, :co, :n] = wt[:, cib * PI: cib * PI + n, :].transpose(2, 0, 1)
    params = []
    bias = np.zeros((L, C), dtype=np.int64)
    for l, layer in enumerate(qm.layers):
        bias[l, : layer.bias.shape[0]] = layer.bias
    mult = np.stack([layer.m for layer in qm.layers[:-1]])
    thr = qm.layers[-1].thr
    lut = np.stack([layer.lut.astype(np.int64) for layer in qm.layers[:-1]])
    for arr in (bias, mult, thr, lut):
        assert np.abs(arr).max() < 2 ** 31, "parameter does not fit int32"
    params = np.concatenate([bias.ravel(), mult.ravel(), thr.ravel(), lut.ravel()]).astype(np.int32)
    return W.view(np.uint8).ravel(), params


def pack_windows(x_int: np.ndarray) -> np.ndarray:
    """x_int (B, 4, T, D, D) with values 0, 1, 2 -> uint8 (B, T * D * D)."""
    B = x_int.shape[0]
    v = x_int.reshape(B, 4, -1).astype(np.uint8)
    return (v[:, 0] | (v[:, 1] << 2) | (v[:, 2] << 4) | (v[:, 3] << 6)).astype(np.uint8)


def unpack_bits(b: np.ndarray, T: int, D: int) -> np.ndarray:
    """uint8 (B, T * D * D) -> bits (B, 4, T, D, D)."""
    B = b.shape[0]
    return np.stack([(b >> k) & 1 for k in range(4)], axis=1).reshape(B, 4, T, D, D).astype(np.uint8)
