#!/usr/bin/env python3
"""Generate the synthetic-ground figures from synthetic_sweep.jsonl.

Reads the committed 320-row dataset and writes PNGs to results/figures/. Every
figure is regenerable from this script alone (no hand-editing, no external state),
so the thesis can cite a figure and the reader can rebuild it byte-for-byte from
the committed rows. Means are over the 8 repeats per cell; error bars are 1 SD
(they make the high low-depth Code Mode variance visible, which is deliberate).

Effective input (caching-fair cost) =
    prompt_tokens*1.0 + cache_creation_tokens*1.25 + cache_read_tokens*0.1.

Run:  ../synthetic/.venv/Scripts/python.exe plot.py
"""
import json
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "synthetic_sweep.jsonl"
FIGS = HERE.parent / "figures"
FIGS.mkdir(exist_ok=True)

DEPTHS = [1, 2, 4, 8]
# colour/marker per arm; cache state distinguished by linestyle in the caching figure
ARM_STYLE = {
    "standard": {"color": "#c1442e", "marker": "o", "label": "standard MCP"},
    "codemode": {"color": "#2e6fc1", "marker": "s", "label": "Code Mode"},
}


def load():
    return [json.loads(l) for l in DATA.read_text(encoding="utf-8").splitlines() if l.strip()]


def eff(r):
    return r["prompt_tokens"] + r["cache_creation_tokens"] * 1.25 + r["cache_read_tokens"] * 0.1


def series(rows, arm, K, cache, fn):
    """mean and SD of fn over repeats, per depth, for one (arm, K, cache) line."""
    ys, es = [], []
    for d in DEPTHS:
        cell = [r for r in rows if r["rq"] == "rq1" and r["surface"] == "coarse"
                and r["tool_count"] == K and r["depth"] == d and r["arm"] == arm
                and r["cache"] == cache and not r["error"]]
        vals = [fn(r) for r in cell]
        ys.append(st.mean(vals) if vals else float("nan"))
        es.append(st.pstdev(vals) if len(vals) > 1 else 0.0)
    return ys, es


def _depth_axis(ax):
    ax.set_xticks(DEPTHS)
    ax.set_xlabel("orchestration depth D (dependent tool calls)")
    ax.grid(True, alpha=0.3)


def two_panel(rows, fn, ylabel, title, fname, scale=1.0):
    """One metric vs depth, cache=off, panel per K, line per arm."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, K in zip(axes, (3, 25)):
        for arm, sty in ARM_STYLE.items():
            ys, es = series(rows, arm, K, "off", fn)
            ys = [y / scale for y in ys]
            es = [e / scale for e in es]
            ax.errorbar(DEPTHS, ys, yerr=es, capsize=3, **{k: sty[k] for k in ("color", "marker")},
                        label=sty["label"], linewidth=2, markersize=6)
        ax.set_title(f"K = {K} tools")
        _depth_axis(ax)
    axes[0].set_ylabel(ylabel)
    axes[0].legend(frameon=False)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = FIGS / fname
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def caching_figure(rows):
    """Effective input vs depth at K=25, all four (arm x cache) lines.
    cache=off solid, cache=on dashed: shows standard's reclaim at depth and the
    small Code Mode penalty."""
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    for arm, sty in ARM_STYLE.items():
        for cache, ls in (("off", "-"), ("on", "--")):
            ys, _ = series(rows, arm, 25, cache, eff)
            ys = [y / 1000 for y in ys]
            ax.plot(DEPTHS, ys, linestyle=ls, color=sty["color"], marker=sty["marker"],
                    linewidth=2, markersize=6,
                    label=f"{sty['label']}, cache {cache}")
    ax.set_ylabel("effective input (1000s of tokens)")
    ax.set_title("Caching effect at K = 25 (solid = off, dashed = on)")
    _depth_axis(ax)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    out = FIGS / "fig_caching_effect_k25.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def success_figure(rows):
    """Success rate vs depth at K=25, cache=off, per arm (reliability)."""
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    for arm, sty in ARM_STYLE.items():
        ys = []
        for d in DEPTHS:
            cell = [r for r in rows if r["rq"] == "rq1" and r["surface"] == "coarse"
                    and r["tool_count"] == 25 and r["depth"] == d and r["arm"] == arm
                    and r["cache"] == "off"]
            ys.append(100 * sum(r["success"] for r in cell) / len(cell) if cell else float("nan"))
        ax.plot(DEPTHS, ys, color=sty["color"], marker=sty["marker"], linewidth=2,
                markersize=6, label=sty["label"])
    ax.set_ylabel("success rate (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Task success vs depth (K = 25, cache off)")
    _depth_axis(ax)
    ax.legend(frameon=False)
    fig.tight_layout()
    out = FIGS / "fig_success_k25.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def overhead_figure(rows):
    """The negative: output tokens vs depth at K=25, cache=off. Output is billed 5x
    input. Code Mode must WRITE a program, so at shallow depth it emits MORE output
    than standard's terse 'call next tool' messages; the advantage only appears once
    depth amortizes that overhead (the lines cross)."""
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    for arm, sty in ARM_STYLE.items():
        ys, es = series(rows, arm, 25, "off", lambda r: r["completion_tokens"])
        ax.errorbar(DEPTHS, ys, yerr=es, capsize=3, color=sty["color"], marker=sty["marker"],
                    linewidth=2, markersize=6, label=sty["label"])
    ax.set_ylabel("output tokens (billed 5x input)")
    ax.set_title("Code Mode's overhead at shallow depth (K = 25, cache off)")
    _depth_axis(ax)
    ax.legend(frameon=False)
    fig.tight_layout()
    out = FIGS / "fig_codemode_overhead_k25.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main():
    rows = load()
    made = []
    made.append(overhead_figure(rows))
    made.append(two_panel(rows, lambda r: r["reasoning_turns"],
                          "reasoning turns (model round trips)",
                          "Model round trips scale with depth for standard MCP, flat for Code Mode",
                          "fig_turns_vs_depth.png"))
    made.append(two_panel(rows, eff,
                          "effective input (1000s of tokens)",
                          "Token cost vs depth (cache off)",
                          "fig_effinput_vs_depth.png", scale=1000.0))
    made.append(two_panel(rows, lambda r: r["latency_total_ms"],
                          "latency (seconds)",
                          "Latency vs depth (cache off)",
                          "fig_latency_vs_depth.png", scale=1000.0))
    made.append(caching_figure(rows))
    made.append(success_figure(rows))
    for p in made:
        print("wrote", p.relative_to(HERE.parent.parent))


if __name__ == "__main__":
    main()
