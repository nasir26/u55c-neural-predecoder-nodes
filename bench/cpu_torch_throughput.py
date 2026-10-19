"""Host CPU baseline: windows per second of NVIDIA's model 1 in PyTorch (eval mode, no gradient), by batch size,
precision and thread count. The values of the weights do not affect the speed.

Author: Nasir Ali, C-DAC Noida.
usage: cpu_torch_throughput.py --checkpoint <safetensors> [--d 9] [--t 9] [--threads 256] [--out file.json]
"""
import argparse, json, os, sys, time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "harness"))
import nvidia_bridge as nb  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--d", type=int, default=9)
    ap.add_argument("--t", type=int, default=9)
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--batches", default="1,16,64,256")
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    model, _ = nb.load_reference_model(a.checkpoint)
    rows = []
    for dtype in (torch.float32, torch.bfloat16):
        m = model.to(dtype)
        for bs in map(int, a.batches.split(",")):
            x = (torch.rand(bs, 4, a.t, a.d, a.d) < 0.03).to(dtype)
            with torch.inference_mode():
                m(x)                                    # warm up
                n, t0 = 0, time.perf_counter()
                while time.perf_counter() - t0 < a.seconds:
                    m(x); n += bs
                dt = time.perf_counter() - t0
            rows.append(dict(dtype=str(dtype).split(".")[-1], batch=bs, windows_per_s=n / dt))
            print(f"{rows[-1]['dtype']:8s} batch {bs:4d}: {n / dt:10.1f} windows/s", flush=True)
    rec = dict(d=a.d, t=a.t, threads=a.threads, torch=torch.__version__, rows=rows)
    if a.out:
        json.dump(rec, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
