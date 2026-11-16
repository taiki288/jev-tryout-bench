#!/usr/bin/env python3
"""Contour plots of bench/latency_grid.py results: time per decision over (image size, number of options).

  python3 bench/grid_plot.py results/contour_*.grid.jsonl --out docs/img/contour.png

One panel per run label, all on the same colour scale. Lines mark 3 Hz (333 ms), 2 Hz (500 ms) and 1 Hz (1 s).
The x axis is the image width (the grids use 2:1 panoramas, so the height is half of it); x = 0 is "no image".
Combinations a model cannot take (e.g. Smaug above 20 options) are left blank.
Needs numpy + matplotlib only (on the Orin the system python3 has both).
"""
import argparse, json, math
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402,F401
import numpy as np  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402

plt.rcParams["font.family"] = ["Noto Sans CJK JP", "Noto Serif CJK JP", "DejaVu Sans"]
LINES = [(333, "3 Hz"), (500, "2 Hz"), (1000, "1 Hz")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", default="contour.png")
    ap.add_argument("--stat", default="mean_ms", choices=["mean_ms", "median_ms", "p90_ms"])
    ap.add_argument("--cols", type=int, default=2)
    ap.add_argument("--lines", type=int, default=20,
                    help="also write <out>_K<lines>.png: time vs image width for this option count, all runs (0 = off)")
    a = ap.parse_args()

    runs = defaultdict(dict)  # label -> (width, options) -> ms
    for f in a.files:
        for line in open(f):
            r = json.loads(line)
            runs[r["label"]][r["width"], r["options"]] = r[a.stat]
    widths = sorted({w for d in runs.values() for w, _ in d})
    opts = sorted({k for d in runs.values() for _, k in d})
    allv = [v for d in runs.values() for v in d.values()]
    norm = LogNorm(vmin=min(allv), vmax=max(allv))
    levels = np.geomspace(min(allv), max(allv), 24)

    n = len(runs); cols = min(a.cols, n); rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(6.2 * cols, 4.9 * rows), squeeze=False,
                             gridspec_kw={"hspace": 0.35})
    for ax, (label, d) in zip(axes.flat, runs.items()):
        Z = np.array([[d.get((w, k), np.nan) for w in widths] for k in opts])
        X, Y = np.meshgrid(widths, opts)
        cf = ax.contourf(X, Y, np.ma.masked_invalid(Z), levels=levels, norm=norm, cmap="viridis")
        cs = ax.contour(X, Y, np.ma.masked_invalid(Z), levels=[v for v, _ in LINES], colors="white", linewidths=1.6)
        ax.clabel(cs, fmt={v: t for v, t in LINES}, fontsize=9)
        ax.scatter(X[~np.isnan(Z)], Y[~np.isnan(Z)], s=4, c="k", alpha=0.3)
        if np.isnan(Z).any():
            ax.text(0.5, 0.75, "この選択肢の数は非対応", transform=ax.transAxes, ha="center", color="0.4")
        ax.set_title(label)
        ax.set_xlabel("画像の横幅 px（横:縦 = 2:1、0 = 画像なし）")
        ax.set_ylabel("選択肢の数")
    for ax in axes.flat[n:]:
        ax.axis("off")
    ticks = [t for t in [50, 100, 200, 333, 500, 1000, 2000, 4000] if norm.vmin <= t <= norm.vmax]
    cb = fig.colorbar(cf, ax=axes.ravel().tolist(), label="1 回の判定の時間 ms（対数）", ticks=ticks)
    cb.ax.set_yticklabels([str(t) for t in ticks]); cb.ax.minorticks_off()
    fig.suptitle("Jetson AGX Orin 64GB: 判定 1 回の時間（白線: 3 / 2 / 1 Hz）")
    fig.savefig(a.out, dpi=130, bbox_inches="tight")
    print("wrote", a.out)

    if a.lines:
        fig, ax = plt.subplots(figsize=(8, 4.8))
        for label, d in runs.items():
            pts = sorted((w, v) for (w, k), v in d.items() if k == a.lines)
            if pts:
                ax.plot([w for w, _ in pts], [v for _, v in pts], marker="o", label=label)
        for v, t in LINES:
            ax.axhline(v, color="0.6", ls="--", lw=1); ax.text(widths[-1], v, " " + t, va="center", color="0.4")
        ax.set_yscale("log"); ax.set_yticks([50, 100, 200, 333, 500, 1000, 2000, 4000])
        ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter()); ax.minorticks_off()
        ax.set_xlabel("画像の横幅 px（横:縦 = 2:1、0 = 画像なし）"); ax.set_ylabel("1 回の判定の時間 ms")
        ax.set_title(f"選択肢 {a.lines} 個のときの時間"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
        out2 = a.out.rsplit(".", 1)[0] + f"_K{a.lines}.png"
        fig.savefig(out2, dpi=130, bbox_inches="tight")
        print("wrote", out2)


if __name__ == "__main__":
    main()
