"""Card campaign of the pre-decoder kernel: bit exact check of every instance of every card against the integer
model, throughput on one, two and four cards (shot parallel, one host process per card, common start barrier), and
latency of one window on one instance.

Author: Nasir Ali, C-DAC Noida.

For each network (NVIDIA model 1, Fast, at d = 9; model 4, Accurate, at d = 13; both 9 or 13 rounds) the integer
model is calibrated on syndrome windows of NVIDIA's inference datapipe (X basis), packed (host/pack.py), and run by
the XRT runtime (host/xrt_predecoder.cpp). Card temperatures (xbutil, read only) are recorded before and after each
throughput point. Output: one JSON record.

usage: run_cards.py --xclbin build/hw/predecoder.xclbin [--cards 0,1,2,3] [--out ../runs/hw/cards.json]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT / "harness"), str(ROOT / "model"), str(ROOT / "host")]
import nvidia_bridge as nb  # noqa: E402
from pack import pack_model, pack_windows, unpack_bits  # noqa: E402
from qmodel import calibrate  # noqa: E402

HOST = Path(os.environ.get("NPD_HOST", nb.RUNS / "bin" / "xrt_predecoder"))
NETS = {  # name: (checkpoint file, NVIDIA model id, distance = rounds)
    "fast": ("ising_decoder_surface_code_1_fast_r9_v1.0.77_fp16.safetensors", 1, 9),
    "accurate": ("ising_decoder_surface_code_1_accurate_r13_v1.0.86_fp16.safetensors", 4, 13),
}


def temps() -> dict:
    """FPGA and HBM temperature of every card (xbutil, read only)."""
    out = {}
    lst = subprocess.run(["xbutil", "examine"], capture_output=True, text=True).stdout
    for bdf in re.findall(r"\[([0-9a-f:.]+)\]", lst):
        r = subprocess.run(["xbutil", "examine", "-d", bdf, "-r", "thermal"], capture_output=True, text=True).stdout
        f = re.search(r"^\s*FPGA\s+(\d+)", r, re.M)
        h = re.search(r"^\s*FPGA HBM\s+(\d+)", r, re.M)
        out[bdf] = dict(fpga_c=int(f.group(1)) if f else None, hbm_c=int(h.group(1)) if h else None)
    return out


def prepare(name: str, d: Path, n_cal: int, n_win: int):
    ckpt, mid, dist = NETS[name]
    cfg = nb.make_cfg(dist, dist, model_id=mid)
    model, _ = nb.load_reference_model(str(nb.RUNS / "models" / "nvidia" / ckpt))
    model = model.float()
    X = torch.stack([b["trainX"] for _, b in zip(range(n_cal + n_win), nb.sample_windows(cfg, "X", n_cal + n_win))])
    qm = calibrate(model, X[:n_cal])
    xi = np.rint(X[n_cal:].numpy() * 2).astype(np.int64)
    want, _ = qm.forward_int_fast(xi)
    w, p = pack_model(qm)
    f = dict(w=d / f"{name}_w.bin", p=d / f"{name}_p.bin", x=d / f"{name}_x.bin")
    w.tofile(f["w"]); p.tofile(f["p"]); pack_windows(xi).tofile(f["x"])
    return dict(files=f, want=want, dist=dist, layers=len(qm.layers), windows=len(xi))


def cmd(xclbin, f, dist, windows, launches, dev, dump=None):
    c = [str(HOST), str(xclbin), str(f["w"]), str(f["p"]), str(f["x"]), str(dist), str(dist), str(windows),
         str(launches), str(dev)]
    return c + ([str(dump)] if dump else [])


def check(xclbin, cards, prep, d, rec, name):
    for dev in cards:
        dump = d / "bits.bin"
        r = subprocess.run(cmd(xclbin, prep["files"], prep["dist"], prep["windows"], 1, dev, dump),
                           capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError(r.stderr)
        n_inst = int(re.search(r"instances (\d+)", r.stderr).group(1))
        b = np.fromfile(dump, dtype=np.uint8).reshape(n_inst, prep["windows"], -1)
        ok = all(np.array_equal(unpack_bits(b[k], prep["dist"], prep["dist"]), prep["want"]) for k in range(n_inst))
        rec["check"].append(dict(net=name, device=dev, instances=n_inst, windows=prep["windows"], bit_exact=bool(ok)))
        print(f"check {name} device {dev} ({n_inst} instances): {'PASS' if ok else 'FAIL'}", flush=True)
        if not ok:
            raise SystemExit("bit exact check failed")


def bench_once(xclbin, prep, devs, windows, launches, d, inflight=2):
    go = d / "go"
    go.unlink(missing_ok=True)
    env = dict(os.environ, STIM_GO_FILE=str(go), NPD_INFLIGHT=str(inflight))
    ps = [subprocess.Popen(cmd(xclbin, prep["files"], prep["dist"], windows, launches, dev), stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, text=True, env=env) for dev in devs]
    n_inst = []
    for p in ps:
        for line in p.stderr:
            m = re.match(r"instances (\d+)", line)
            if m:
                n_inst.append(int(m.group(1)))
            if line.strip() == "ready":
                break
        else:
            raise RuntimeError(f"host process ended before ready: {p.wait()}")
    go.touch()
    rows = []
    for dev, p in zip(devs, ps):
        rest, rate = p.stderr.read(), p.stdout.read().strip()
        if p.wait() != 0:
            raise RuntimeError(rest)
        m = re.search(r"window_ns (\d+) (\d+) device (\d+) bdf (\S+)", rest)
        rows.append(dict(device=dev, bdf=m.group(4), start_ns=int(m.group(1)), end_ns=int(m.group(2)),
                         card_windows_per_sec=float(rate)))
    t0, t1 = min(r["start_ns"] for r in rows), max(r["end_ns"] for r in rows)
    n = sum(n_inst) * launches * windows
    return dict(cards=len(devs), instances_per_card=n_inst, windows_per_launch=windows, launches=launches,
                inflight=inflight, windows=n, window_s=(t1 - t0) * 1e-9,
                aggregate_windows_per_sec=n / ((t1 - t0) * 1e-9), per_card=rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xclbin", required=True)
    ap.add_argument("--cards", default="0,1,2,3")
    ap.add_argument("--nets", default="fast,accurate")
    ap.add_argument("--cal", type=int, default=256)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--seconds", type=float, default=20.0, help="target duration of one throughput point")
    ap.add_argument("--out", default=str(nb.RUNS / "hw" / f"cards_{time.strftime('%Y%m%d_%H%M')}.json"))
    a = ap.parse_args()
    cards = [int(c) for c in a.cards.split(",")]
    xclbin = Path(a.xclbin).resolve()
    sha = subprocess.run(["sha256sum", str(xclbin)], capture_output=True, text=True).stdout.split()[0]
    rec = dict(start=time.strftime("%Y-%m-%d %H:%M:%S"), xclbin=str(xclbin.name), xclbin_sha256=sha, cards=cards,
               check=[], bench=[], latency=[], temps=[])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        for name in a.nets.split(","):
            prep = prepare(name, d, a.cal, 8)
            rec.setdefault("nets", {})[name] = dict(checkpoint=NETS[name][0], model_id=NETS[name][1],
                                                     distance=prep["dist"], rounds=prep["dist"], layers=prep["layers"])
            check(xclbin, cards, prep, d, rec, name)
            # size a launch so one throughput point lasts about --seconds on one card (probe: 8 windows, 4 launches)
            probe = bench_once(xclbin, prep, cards[:1], 8, 4, d)
            rate = probe["aggregate_windows_per_sec"]
            windows = 64
            launches = max(4, int(a.seconds * rate / (windows * sum(probe["instances_per_card"]))))
            for n in (1, 2, 4):
                if n > len(cards):
                    continue
                for rep in range(a.reps):
                    rec["temps"].append(dict(when="before", net=name, cards=n, rep=rep, t=time.time(), temps=temps()))
                    r = bench_once(xclbin, prep, cards[:n], windows, launches, d)
                    r.update(net=name, rep=rep)
                    rec["bench"].append(r)
                    rec["temps"].append(dict(when="after", net=name, cards=n, rep=rep, t=time.time(), temps=temps()))
                    print(f"bench {name} {n} card(s) rep {rep}: {r['aggregate_windows_per_sec']:.1f} windows/s",
                          flush=True)
            # latency: one window per launch, one launch in flight, one card
            r = bench_once(xclbin, prep, cards[:1], 1, 200, d, inflight=1)
            r.update(net=name, per_window_s=r["window_s"] / r["launches"])
            rec["latency"].append(r)
            print(f"latency {name}: {r['per_window_s'] * 1e3:.3f} ms per window (one in flight)", flush=True)
            json.dump(rec, open(a.out, "w"), indent=1)
    rec["end"] = time.strftime("%Y-%m-%d %H:%M:%S")
    json.dump(rec, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
