"""Integer model of NVIDIA's surface code pre-decoder (model 1: four 3x3x3 convolutions, 4 -> 128 -> 128 -> 128 -> 4,
GELU with tanh approximation), the exact arithmetic the FPGA kernel implements.

Author: Nasir Ali, C-DAC Noida.

Arithmetic, layer by layer:
  input     x_int = 2 * trainX, values 0, 1, 2 (channels 0 and 1 are syndrome bits, 2 and 3 fixed geometry masks with
            values 0, 0.5, 1), scale 1/2
  layer l   acc = sum over 27 neighbours and input channels of w_int8 * a_int  (int32, zero padding)  + bias_int32
            hidden layers: p = clamp(round_shift(acc * M_o, S_o), -2^(B-1), 2^(B-1) - 1) per output channel o,
            then a = LUT[p], an int8 activation of the GELU output at the layer's activation scale
            last layer: bit = acc >= T_c, with T_c the integer form of the logit threshold of channel c
  round_shift(v, S) = (v + 2^(S-1)) >> S  (arithmetic shift, round half up)
Weights are symmetric per output channel; activations are symmetric with zero point 0, so the zero padding of the
float model is exactly an integer 0. Every scale is fixed by calibrate() from float activations of sample windows.

The model is a torch.nn.Module whose forward returns float "logits" whose sign relative to the threshold equals the
integer decision, so NVIDIA's evaluation code can use it in place of the float model unchanged.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch

PRE_BITS = 8          # width of the requantized pre activation that indexes the GELU table
ACC_SHIFT = 24        # fixed point precision of the requantization multipliers


def gelu_tanh(x):
    return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))


def round_shift(v: np.ndarray, s: int) -> np.ndarray:
    return (v + (np.int64(1) << (s - 1))) >> s


@dataclass
class QLayer:
    w: np.ndarray                 # int8 (out, in, 3, 3, 3)
    bias: np.ndarray              # int32 (out,), in accumulator units
    m: np.ndarray = None          # int64 (out,), requantization multiplier (hidden layers)
    shift: int = ACC_SHIFT
    lut: np.ndarray = None        # int8 (2^PRE_BITS,), GELU table indexed by p + 2^(PRE_BITS-1)
    thr: np.ndarray = None        # int64 (out,), integer thresholds (last layer)
    meta: dict = field(default_factory=dict)


def conv3d_int(a: np.ndarray, w: np.ndarray) -> np.ndarray:
    """a: (B, Cin, T, D, D) int, w: (Cout, Cin, 3, 3, 3) int8 -> (B, Cout, T, D, D) int64, zero padding 1."""
    B, C, T, H, W = a.shape
    p = np.zeros((B, C, T + 2, H + 2, W + 2), dtype=np.int64)
    p[:, :, 1:-1, 1:-1, 1:-1] = a
    out = np.zeros((B, w.shape[0], T, H, W), dtype=np.int64)
    w64 = w.astype(np.int64)
    for dt in range(3):
        for dh in range(3):
            for dw in range(3):
                patch = p[:, :, dt:dt + T, dh:dh + H, dw:dw + W]          # (B, C, T, H, W)
                out += np.einsum("bcthw,oc->bothw", patch, w64[:, :, dt, dh, dw], optimize=True)
    return out


class QuantizedPreDecoder(torch.nn.Module):
    def __init__(self, layers: list[QLayer], out_scale: np.ndarray, thresholds: tuple[float, float]):
        super().__init__()
        # unused placeholder: NVIDIA's evaluation module reads next(self.parameters()) to find the device
        self.device_anchor = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.layers = layers
        self.out_scale = out_scale            # float (4,), accumulator units -> logits (for reporting only)
        self.thresholds = thresholds

    # ---------------------------------------------------------------- integer forward
    def forward_int(self, x_int: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """x_int (B, 4, T, D, D) in {0, 1, 2}. Returns (bits (B, 4, T, D, D) uint8, last layer accumulators)."""
        a = x_int.astype(np.int64)
        for i, L in enumerate(self.layers):
            acc = conv3d_int(a, L.w) + L.bias.astype(np.int64)[None, :, None, None, None]
            if L.lut is not None:
                lo, hi = -(1 << (PRE_BITS - 1)), (1 << (PRE_BITS - 1)) - 1
                p = np.clip(round_shift(acc * L.m[None, :, None, None, None], L.shift), lo, hi)
                a = L.lut[p - lo].astype(np.int64)
            else:
                bits = (acc >= L.thr[None, :, None, None, None]).astype(np.uint8)
                return bits, acc
        raise RuntimeError("last layer must carry thresholds")

    @torch.no_grad()
    def forward_int_fast(self, x_int: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Same integers as forward_int, with the convolutions in float64 PyTorch: every accumulator is an integer
        below 2^27 in magnitude (128 x 27 x 127 x 127 plus bias), far inside the 2^53 range float64 represents
        exactly, so the result is bit identical (checked by tests/test_qmodel.py)."""
        a = torch.from_numpy(x_int.astype(np.float64))
        for L in self.layers:
            w = torch.from_numpy(L.w.astype(np.float64))
            acc = torch.nn.functional.conv3d(a, w, padding=1).numpy().astype(np.int64)
            acc += L.bias.astype(np.int64)[None, :, None, None, None]
            if L.lut is not None:
                lo, hi = -(1 << (PRE_BITS - 1)), (1 << (PRE_BITS - 1)) - 1
                p = np.clip(round_shift(acc * L.m[None, :, None, None, None], L.shift), lo, hi)
                a = torch.from_numpy(L.lut[p - lo].astype(np.float64))
            else:
                return (acc >= L.thr[None, :, None, None, None]).astype(np.uint8), acc
        raise RuntimeError("last layer must carry thresholds")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_int = np.rint(x.detach().cpu().numpy().astype(np.float64) * 2.0).astype(np.int64)
        bits, acc = self.forward_int_fast(x_int)
        th = np.array([self.thresholds[0], self.thresholds[0], self.thresholds[1], self.thresholds[1]])
        # +-1 around the threshold: NVIDIA's code thresholds these exactly as the integer decision
        logits = np.where(bits == 1, th[None, :, None, None, None] + 1.0, th[None, :, None, None, None] - 1.0)
        return torch.from_numpy(logits.astype(np.float32))


# -------------------------------------------------------------------- calibration
@torch.no_grad()
def float_activations(model: torch.nn.Module, x: torch.Tensor) -> list[torch.Tensor]:
    """Pre activations of every convolution of NVIDIA's PreDecoderModelMemory_v1 (float32, eval mode)."""
    convs = [m for m in model.net if isinstance(m, torch.nn.Conv3d)]
    outs, a = [], x.float()
    for i, c in enumerate(convs):
        z = torch.nn.functional.conv3d(a, c.weight.float(), c.bias.float(), padding=1)
        outs.append(z)
        a = torch.nn.functional.gelu(z, approximate="tanh")
    return outs


@torch.no_grad()
def calibrate(model: torch.nn.Module, x_cal: torch.Tensor, thresholds=(0.0, 0.0), percentile=99.99) -> QuantizedPreDecoder:
    """Integer model from a float model and calibration inputs (B, 4, T, D, D)."""
    convs = [m for m in model.net if isinstance(m, torch.nn.Conv3d)]
    pre = float_activations(model, x_cal)
    layers = []
    s_in = 0.5                                    # input scale: x = s_in * x_int
    for i, c in enumerate(convs):
        w = c.weight.float().numpy().astype(np.float64)
        b = c.bias.float().numpy().astype(np.float64)
        sw = np.maximum(np.abs(w).reshape(w.shape[0], -1).max(axis=1), 1e-12) / 127.0     # per output channel
        w_int = np.clip(np.rint(w / sw[:, None, None, None, None]), -127, 127).astype(np.int8)
        s_acc = sw * s_in                                                                     # accumulator scale
        bias = np.rint(b / s_acc).astype(np.int64)
        if i < len(convs) - 1:
            z = pre[i].numpy()
            s_pre = np.percentile(np.abs(z), percentile) / ((1 << (PRE_BITS - 1)) - 1)       # pre activation scale
            m = np.rint(s_acc / s_pre * (1 << ACC_SHIFT)).astype(np.int64)
            g = gelu_tanh(z)
            s_act = max(np.percentile(np.abs(g), percentile), 1e-12) / 127.0                  # activation scale
            idx = np.arange(-(1 << (PRE_BITS - 1)), 1 << (PRE_BITS - 1))
            lut = np.clip(np.rint(gelu_tanh(idx * s_pre) / s_act), -127, 127).astype(np.int8)
            layers.append(QLayer(w_int, bias, m=m, lut=lut,
                                 meta=dict(s_w=sw, s_in=s_in, s_pre=s_pre, s_act=s_act)))
            s_in = s_act
        else:
            th = np.array([thresholds[0], thresholds[0], thresholds[1], thresholds[1]], dtype=np.float64)
            thr = np.ceil(th / s_acc).astype(np.int64)   # acc * s_acc >= th  <=>  acc >= ceil(th / s_acc)
            layers.append(QLayer(w_int, bias, thr=thr, meta=dict(s_w=sw, s_in=s_in)))
            out_scale = s_acc
    return QuantizedPreDecoder(layers, out_scale, thresholds)
