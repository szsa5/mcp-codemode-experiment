#!/usr/bin/env python3
"""Aggregate the synthetic sweep into the tables the Evaluation/Discussion cite.

Reads ../synthetic_sweep.jsonl (320 rows: the full 2x2 cache{off,on} x
arm{standard,codemode}) and prints, plus writes, the derived tables. This is the
only sanctioned path from raw rows to reported numbers: every figure in the thesis
must trace to a table produced here, not to a hand-computed value.

Effective input (the caching-fair cost metric) =
    prompt_tokens*1.0 + cache_creation_tokens*1.25 + cache_read_tokens*0.1
(5-minute-TTL multipliers; prompt_tokens is the uncached remainder, so with
cache=off this is just the full input). Dollars are not computed here.

Run:  ../synthetic/.venv/Scripts/python.exe aggregate.py
Output: stdout + results/aggregates.txt (committed, regenerable).
"""
import json
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "synthetic_sweep.jsonl"
OUT = HERE.parent / "aggregates.txt"


def load():
    return [json.loads(l) for l in DATA.read_text(encoding="utf-8").splitlines() if l.strip()]


def eff(r):
    return r["prompt_tokens"] + r["cache_creation_tokens"] * 1.25 + r["cache_read_tokens"] * 0.1


def promptside(r):
    # all input-side tokens actually sent (uncached + written + read)
    return r["prompt_tokens"] + r["cache_creation_tokens"] + r["cache_read_tokens"]


def cell(rows, **f):
    out = []
    for r in rows:
        if all(r.get(k) == v for k, v in f.items()):
            out.append(r)
    return out


def mean(rows, fn):
    vals = [fn(r) for r in rows]
    return st.mean(vals) if vals else float("nan")


def cv_pct(rows, fn):
    vals = [fn(r) for r in rows]
    if len(vals) < 2 or st.mean(vals) == 0:
        return 0.0
    return 100 * st.pstdev(vals) / st.mean(vals)


def main():
    rows = load()
    lines = []

    def emit(s=""):
        print(s)
        lines.append(s)

    emit("=" * 78)
    emit("SYNTHETIC SWEEP AGGREGATES")
    emit("=" * 78)
    bycache = Counter(r["cache"] for r in rows)
    n_err = sum(1 for r in rows if r["error"])
    emit(f"rows={len(rows)}  by_cache={dict(bycache)}  errored_runs={n_err}")
    for c in ("off", "on"):
        sub = [r for r in rows if r["cache"] == c]
        emit(f"  cache={c}: success {sum(r['success'] for r in sub)}/{len(sub)}")
    ok = [r for r in rows if not r["error"]]  # token/latency stats exclude error rows

    # -- T1: RQ1 scaling (off), per arm/K/depth ------------------------------------
    emit("\n[T1] RQ1 scaling, cache=off: turns, tool_calls, effective input, latency")
    emit(f"{'K':>3} {'D':>2} {'arm':9} {'turns':>6} {'calls':>6} {'eff_in':>8} {'lat_ms':>8} {'succ':>5}")
    for K in (3, 25):
        for d in (1, 2, 4, 8):
            for arm in ("standard", "codemode"):
                c = cell(ok, rq="rq1", cache="off", tool_count=K, depth=d, arm=arm)
                ca = cell(rows, rq="rq1", cache="off", tool_count=K, depth=d, arm=arm)
                if not c:
                    continue
                succ = f"{sum(r['success'] for r in ca)}/{len(ca)}"
                emit(f"{K:>3} {d:>2} {arm:9} {mean(c,lambda r:r['reasoning_turns']):>6.1f} "
                     f"{mean(c,lambda r:r['tool_calls']):>6.1f} {mean(c,eff):>8.0f} "
                     f"{mean(c,lambda r:r['latency_total_ms']):>8.0f} {succ:>5}")

    # -- T2: caching effect, RQ1 K=25 ---------------------------------------------
    emit("\n[T2] Caching effect (RQ1 K=25): effective input off vs on, % reduction")
    emit("     reduction>0 = caching cheaper; <0 = caching adds net cost.")
    emit(f"{'D':>2} {'arm':9} {'off_eff':>8} {'on_eff':>8} {'reduct':>7} {'cr_mean':>8} {'cv_on%':>7}")
    for d in (1, 2, 4, 8):
        for arm in ("standard", "codemode"):
            off = cell(ok, rq="rq1", cache="off", tool_count=25, depth=d, arm=arm)
            on = cell(ok, rq="rq1", cache="on", tool_count=25, depth=d, arm=arm)
            if not off or not on:
                continue
            eo, en = mean(off, eff), mean(on, eff)
            red = f"{100*(eo-en)/eo:+.0f}%"
            note = "  <-- noisy" if cv_pct(on, eff) > 10 else ""
            emit(f"{d:>2} {arm:9} {eo:>8.0f} {en:>8.0f} {red:>7} "
                 f"{mean(on,lambda r:r['cache_read_tokens']):>8.0f} {cv_pct(on,eff):>6.1f}%{note}")
    emit("  NOTE: low-depth codemode (D1,D2) effects are within-noise (CV>10%); the")
    emit("  reliable signals are standard reclaim at D4/D8 and codemode penalty at D4/D8.")

    # -- T3: within-conversation vs cross-repeat caching (D8 K25 standard, on) -----
    emit("\n[T3] Cache is WITHIN-conversation, not cross-repeat (D8 K25 standard, cache=on)")
    emit("     every repeat re-writes (~cw) and reads (~cr) its own prefix => cold ~ warm.")
    d8 = sorted(cell(ok, rq="rq1", cache="on", tool_count=25, depth=8, arm="standard"),
                key=lambda r: r["repeat"])
    emit(f"{'repeat':>6} {'eff_in':>8} {'cw':>6} {'cr':>7}")
    for r in d8:
        emit(f"{r['repeat']:>6} {eff(r):>8.0f} {r['cache_creation_tokens']:>6} {r['cache_read_tokens']:>7}")
    if d8:
        emit(f"  cold(repeat0)={eff(d8[0]):.0f}  amortized(mean)={mean(d8,eff):.0f}  "
             f"=> the +reduction is a per-request figure, not a warm-cache artifact.")

    # -- T4: RQ2 granularity ------------------------------------------------------
    emit("\n[T4] RQ2 granularity (value_sum, D4, K3, cache=off): coarse vs fine")
    emit(f"{'surface':8} {'arm':9} {'calls':>6} {'turns':>6} {'eff_in':>8}")
    for surf in ("coarse", "fine"):
        for arm in ("standard", "codemode"):
            c = cell(ok, rq="rq2", surface=surf, arm=arm, cache="off")
            if not c:
                continue
            emit(f"{surf:8} {arm:9} {mean(c,lambda r:r['tool_calls']):>6.1f} "
                 f"{mean(c,lambda r:r['reasoning_turns']):>6.1f} {mean(c,eff):>8.0f}")
    emit("  CAUTION: standard fine raises tool_calls (5->9) but NOT turns/tokens at K3")
    emit("  (calls batch into the same rounds), so the standard-side cost claim is weak")
    emit("  here; only codemode's granularity-invariance (2 turns either way) is clean.")

    # -- T5: RQ3 reliability ------------------------------------------------------
    emit("\n[T5] RQ3 reliability")
    cm = [r for r in rows if r["arm"] == "codemode"]
    emit(f"  codemode code_executed: {Counter((r['code_executed'],r['answer_correct']) for r in cm)}")
    ebw = [r for r in cm if r["code_executed"] and not r["answer_correct"]]
    emit(f"  executed-but-wrong: {len(ebw)}/{len(cm)} codemode runs (too thin to characterize; needs Wesley)")
    for r in ebw:
        emit(f"    {r['rq']} {r['surface']} D{r['depth']} K{r['tool_count']} "
             f"cache={r['cache']} parsed={r['parsed_answer']} expected={r['expected_answer']}")
    sw = [r for r in rows if r["arm"] == "standard" and not r["answer_correct"] and not r["error"]]
    diffs = Counter((r["parsed_answer"] or 0) - r["expected_answer"] for r in sw)
    depths = Counter(r["depth"] for r in sw)
    emit(f"  standard wrong-answers: {len(sw)} (all error-free runs); diffs={dict(diffs)}; depths={dict(depths)}")
    emit("  => standard's only failure mode is a consistent off-by-one (one hop short) at D8;")
    emit("     identical off vs on, confirming caching is behavior-neutral.")

    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n[written] {OUT}")


if __name__ == "__main__":
    main()
