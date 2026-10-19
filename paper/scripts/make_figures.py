#!/usr/bin/env python3
"""Figures of the manuscript (figures/*.pdf).

Author: Nasir Ali, C-DAC Noida.

fig_kernel: block diagram of one kernel instance. fig_throughput: measured windows per second on one, two and four
cards and on the host CPU (latest runs/hw/cards_*.json, runs/cpu/cpu_*.json). fig_ler: improvement factor of the
pre-decoder over PyMatching alone, ours against NVIDIA's published values (runs/ler/*.json). Missing data gives a
placeholder text; the final build refuses placeholders through the PENDING check of the macros.
usage: make_figures.py
"""
import glob
import json
import os
import statistics
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.abspath(os.path.join(HERE, ".."))
RUNS = os.environ.get("NPD_RUNS", os.path.join(PAPER, "..", "results"))
FIG = os.path.join(PAPER, "figures")
sys.path.insert(0, HERE)
from make_numbers import NVIDIA, ler  # noqa: E402

plt.rcParams.update({"font.size": 9, "font.family": "serif", "pdf.fonttype": 42})
BLUE, ORANGE, GREEN, GREY = "#2b4a7a", "#c8642a", "#3a8a5c", "#777777"


def save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, name + ".pdf"), bbox_inches="tight", metadata={"CreationDate": None})
    plt.close(fig)


def box(ax, x, y, w, h, text, fc="#e8eef7", ec=BLUE, fs=8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=ec, lw=1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)


def arrow(ax, x0, y0, x1, y1, text=None):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", lw=0.9, color="#333333"))
    if text:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.12, text, fontsize=7, ha="center", va="bottom", color="#333333")


def fig_kernel():
    fig, ax = plt.subplots(figsize=(7.2, 3.1))
    ax.set_xlim(0, 12); ax.set_ylim(0, 4.6); ax.axis("off")
    box(ax, 0.1, 3.0, 2.0, 1.2, "HBM\nweights and\nparameters", fc="#f2f2f2", ec=GREY, fs=7)
    box(ax, 0.1, 0.5, 2.0, 1.2, "HBM\nwindows in,\nbits out", fc="#f2f2f2", ec=GREY, fs=7)
    box(ax, 3.0, 3.0, 2.3, 1.2, "weights (URAM),\nsix layers;\nparameters", fc="#eef5ee", ec=GREEN, fs=7)
    box(ax, 3.0, 0.5, 2.3, 1.2, "activations:\ntwo buffers,\n128 channels, 8 bit", fc="#eef5ee", ec=GREEN, fs=7)
    box(ax, 6.1, 1.75, 2.7, 1.6, "convolution engine\n64 \u00d7 16 weight block,\n1024 products per cycle,\none site per cycle", fs=7)
    box(ax, 6.1, 0.1, 2.7, 1.1, "partial sums: 64 channels\n\u00d7 all sites, 32 bit", fc="#eef5ee", ec=GREEN, fs=7)
    box(ax, 9.5, 1.75, 2.4, 1.6, "epilogue: bias,\nrequantize, GELU\ntable, or threshold\nto bits", fc="#fbefe6", ec=ORANGE, fs=7)
    arrow(ax, 2.1, 3.6, 3.0, 3.6)
    ax.text(1.1, 2.75, "read once per launch", fontsize=6.5, ha="center", va="top", color="#333333")
    arrow(ax, 2.1, 1.1, 3.0, 1.1)
    arrow(ax, 5.3, 3.3, 6.1, 2.95)
    arrow(ax, 5.3, 1.4, 6.1, 2.1)
    ax.annotate("", xy=(7.45, 1.2), xytext=(7.45, 1.75), arrowprops=dict(arrowstyle="<->", lw=0.9, color="#333333"))
    arrow(ax, 8.8, 2.55, 9.5, 2.55)
    ax.annotate("", xy=(4.15, 0.5), xytext=(10.7, 1.75),
                arrowprops=dict(arrowstyle="->", lw=0.9, color=ORANGE, connectionstyle="arc3,rad=-0.3"))
    ax.text(10.7, 0.75, "next layer\nor output bits", fontsize=6.5, color=ORANGE, ha="center")
    save(fig, "fig_kernel")


def fig_throughput():
    files = sorted(glob.glob(os.path.join(RUNS, "hw", "cards_*.json")))
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7))
    for ax, (net, label, cpu_file) in zip(axes, (("fast", "Fast network, $d=T=9$", "cpu_fast_d9.json"),
                                                   ("accurate", "Accurate network, $d=T=13$", "cpu_accurate_d13.json"))):
        names, vals, cols = [], [], []
        p = os.path.join(RUNS, "cpu", cpu_file)
        if os.path.exists(p):
            d = json.load(open(p))
            best = max((r for r in d["rows"] if r["dtype"] == "float32"), key=lambda r: r["windows_per_s"])
            names.append(f"CPU\n{best.get('threads', d['threads'])} threads"); vals.append(best["windows_per_s"])
            cols.append(GREY)
        if files:
            d = json.load(open(files[-1]))
            for n in (1, 2, 4):
                v = [b["aggregate_windows_per_sec"] for b in d["bench"] if b["net"] == net and b["cards"] == n]
                if v:
                    names.append(f"{n} card{'s' if n > 1 else ''}"); vals.append(statistics.median(v)); cols.append(BLUE)
        if not vals:
            ax.text(0.5, 0.5, "measurement pending", ha="center", va="center", transform=ax.transAxes)
        else:
            ax.bar(range(len(vals)), vals, color=cols)
            ax.set_xticks(range(len(vals))); ax.set_xticklabels(names)
            for i, v in enumerate(vals):
                ax.text(i, v, f"{v:,.0f}", ha="center", va="bottom", fontsize=7)
            ax.set_ylim(0, max(vals) * 1.18)
        ax.set_title(label, fontsize=9); ax.set_ylabel("windows per second")
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save(fig, "fig_throughput")


def fig_ler():
    best = ler()
    keys = [("fast", 0.003, 9), ("fast", 0.003, 13), ("fast", 0.006, 9), ("accurate", 0.003, 13), ("accurate", 0.006, 13)]
    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    x = list(range(len(keys)))
    ours = [best.get((n, "fp32", d, p)) for n, p, d in keys]
    ints = [best.get((n, "int8", d, p)) for n, p, d in keys]
    ax.bar([i - 0.27 for i in x], [NVIDIA[k] for k in keys], width=0.27, color=GREY, label="NVIDIA, published (X)")
    def err(rs):  # one standard deviation from counting statistics, the two counts treated as independent
        return [r["ratio_x"] * (1 / r["x_pm"] + 1 / r["x_after"]) ** 0.5 if r else 0 for r in rs]
    eb = dict(ecolor="#333333", capsize=2, error_kw=dict(lw=0.7))
    ax.bar(x, [r["ratio_x"] if r else 0 for r in ours], yerr=err(ours), width=0.27, color=BLUE,
           label="this work, FP32 (X)", **eb)
    ax.bar([i + 0.27 for i in x], [r["ratio_x"] if r else 0 for r in ints], yerr=err(ints), width=0.27, color=ORANGE,
           label="this work, INT8 (X)", **eb)
    ax.axhline(1.0, color="#333333", lw=0.8, ls=":")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{'Fast' if n == 'fast' else 'Accurate'}\n$p={p}$\n$d={d}$" for n, p, d in keys])
    ax.set_ylabel("PyMatching errors /\nerrors with pre-decoder")
    ax.legend(fontsize=7, frameon=False, ncol=3, loc="upper left")
    ax.set_ylim(0, 2.3)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "fig_ler")


if __name__ == "__main__":
    fig_kernel()
    fig_throughput()
    fig_ler()
    print("figures written")
