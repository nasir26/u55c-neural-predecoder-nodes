"""Logical error rate of PyMatching alone and after a pre-decoder variant, through NVIDIA's evaluation code.

Author: Nasir Ali, C-DAC Noida.

Variants: fp16 (NVIDIA's checkpoint as published), fp32 (same weights in float32), int8 (model/qmodel.py, calibrated
on windows drawn separately from the test shots). Every variant is evaluated by
evaluation.logical_error_rate.count_logical_errors_with_errorbar with the same configuration and shot count.
usage: eval_ler.py --checkpoint <safetensors> --variant fp16|fp32|int8 [--d 9] [--t 9] [--shots 65536]
                   [--cal-windows 1024] [--out results.json]
"""
import argparse, json, os, sys, time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "model")]
import nvidia_bridge as nb  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--variant", choices=["fp16", "fp32", "int8"], required=True)
    ap.add_argument("--d", type=int, default=9)
    ap.add_argument("--t", type=int, default=9)
    ap.add_argument("--shots", type=int, default=65536)
    ap.add_argument("--cal-windows", type=int, default=1024)
    ap.add_argument("--percentile", type=float, default=99.99)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    cfg = nb.make_cfg(a.d, a.t)
    dist = nb.dist_manager()
    model, meta = nb.load_reference_model(a.checkpoint)
    info = dict(variant=a.variant, d=a.d, t=a.t, shots_per_basis=a.shots, checkpoint=os.path.basename(a.checkpoint),
                metadata=meta, start=time.strftime("%Y-%m-%d %H:%M:%S"))
    if a.variant == "fp32":
        model = model.float()
    elif a.variant == "int8":
        from qmodel import calibrate
        model = model.float()
        cal = []
        for basis in ("X", "Z"):
            cal += [b["trainX"] for _, b in zip(range(a.cal_windows // 2), nb.sample_windows(cfg, basis, a.cal_windows // 2))]
        model = calibrate(model, torch.stack(cal), percentile=a.percentile)
        info.update(calibration_windows=len(cal), percentile=a.percentile)
    from omegaconf import open_dict
    from evaluation.logical_error_rate import count_logical_errors_with_errorbar
    with open_dict(cfg):
        cfg.test.num_samples = a.shots
        cfg.test.latency_num_samples = 0
        cfg.test.meas_basis_test = "both"
    t0 = time.time()
    res = count_logical_errors_with_errorbar(model, dist.device, dist, cfg)
    info.update(seconds=time.time() - t0, end=time.strftime("%Y-%m-%d %H:%M:%S"))
    out = {}
    for basis, r in (res or {}).items():
        if isinstance(r, dict):
            out[basis] = {k: (float(v) if isinstance(v, (int, float)) else str(v)) for k, v in r.items()
                          if not hasattr(v, "shape")}
    info["result"] = out
    print(json.dumps({b: {k: v for k, v in r.items() if "logical error ratio" in k} for b, r in out.items()}, indent=1))
    if a.out:
        json.dump(info, open(a.out, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
