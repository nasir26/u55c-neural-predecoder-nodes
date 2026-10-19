"""Host CPU baseline: windows per second of NVIDIA's pre-decoder (model 1 or 4) in PyTorch (eval mode, no gradient), by batch size,
precision and thread count, each point repeated --reps times. The values of the weights do not affect the speed.

Author: Nasir Ali, C-DAC Noida.
usage: cpu_torch_throughput.py --checkpoint <safetensors> [--d 9] [--t 9] [--threads 32,64,128,256] [--reps 2]
       [--out file.json]
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
    ap.add_argument("--threads", default=str(os.cpu_count()), help="comma separated thread counts")
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--batches", default="1,4,16,64,256")
    ap.add_argument("--dtypes", default="float32,bfloat16")
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    model, _ = nb.load_reference_model(a.checkpoint)
    rows = []
    for dtype, th, rep in ((getattr(torch, t), int(th), r) for t in a.dtypes.split(",")
                           for th in a.threads.split(",") for r in range(a.reps)):
        torch.set_num_threads(th)
        m = model.to(dtype)
        for bs in map(int, a.batches.split(",")):
            x = (torch.rand(bs, 4, a.t, a.d, a.d) < 0.03).to(dtype)
            with torch.inference_mode():
                m(x)                                    # warm up
                n, t0 = 0, time.perf_counter()
                while time.perf_counter() - t0 < a.seconds:
                    m(x); n += bs
                dt = time.perf_counter() - t0
            rows.append(dict(dtype=str(dtype).split(".")[-1], threads=th, rep=rep, batch=bs, windows_per_s=n / dt))
            print(f"{rows[-1]['dtype']:8s} threads {th:3d} rep {rep} batch {bs:4d}: {n / dt:10.1f} windows/s", flush=True)
    rec = dict(checkpoint=os.path.basename(a.checkpoint), d=a.d, t=a.t, threads=[int(t) for t in a.threads.split(",")], cpu_count=os.cpu_count(), torch=torch.__version__, rows=rows)
    if a.out:
        json.dump(rec, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
