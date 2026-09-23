#!/usr/bin/env python3
"""
compare_judges.py — cross-judge comparison of the FLOP-based robustness metrics.

Joins cost_summary_metrics.csv (target + attacker + judge) and
cost_summary_metrics_nojudge.csv (target + attacker) across judge trees and reports,
per attack, whether the conclusions drawn under the incumbent llama3.1-8b judge survive
under the alternatives.

WHICH AXIS MEANS WHAT
  flops        target + attacker + judge. Moves across judges for TWO reasons at once:
               the judge's own params_b changed (8.03B -> 7.30B -> 3.88B), and the judge
               labelled the responses differently so the risk curve moved. Not a clean
               comparison of conclusions.
  flops_nojudge  target + attacker only. Invariant to the judge's size, so any movement
               is purely the judge's LABELLING changing the risk curve. This is the axis
               to compare conclusions on.

The script reports both, plus the rank correlation of each alternative judge against the
incumbent (does the ordering of models survive?) and the judge's share of total FLOPs
(how much of the accounted compute is the instrument rather than the attack?).

Usage:
    python scripts/compare_judges.py --benchmark harmbench
    python scripts/compare_judges.py --benchmark jailbreakbench --attacks gcg pair jailbroken
    python scripts/compare_judges.py --benchmark harmbench --out-csv /tmp/judge_compare.csv
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

try:
    from scipy.stats import spearmanr
except ImportError:  # scipy is optional; fall back to pandas' rank correlation
    spearmanr = None

# model_id of the incumbent judge; its results live at the top of the tree, not under judges/.
INCUMBENT = "llama3.1-8b-instruct"

MAIN_MODELS = [
    "tulu3-8b-base", "tulu3-8b-sft", "tulu3-8b-dpo", "tulu3-8b-rlvr",
    "qwen2.5-0.5b-instruct", "qwen2.5-3b-instruct", "qwen2.5-7b-instruct",
    "qwen3-4b", "qwen3-4b-saferl",
]


def resolve_scratch(explicit: str | None) -> Path:
    """Find the tree that actually holds rup/.

    setup/start_env.sh exports SCRATCH=/home/<user>/scratch/<user>, which on this cluster is
    one level DEEPER than the ambient $SCRATCH=/scratch/<user>. A shell that never sourced
    start_env.sh therefore points one directory too high and finds nothing. Try the obvious
    candidates and pick the first that actually contains rup/.
    """
    user = os.environ.get("USER", "")
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    if os.environ.get("SCRATCH"):
        base = Path(os.environ["SCRATCH"])
        candidates += [base, base / user]
    if user:
        candidates.append(Path.home() / "scratch" / user)

    for c in candidates:
        if (c / "rup").is_dir():
            return c
    tried = "\n  ".join(str(c) for c in candidates) or "  (none — $SCRATCH unset)"
    raise SystemExit(
        f"Could not find a directory containing rup/. Tried:\n  {tried}\n"
        "Pass --scratch explicitly, e.g. --scratch /home/$USER/scratch/$USER"
    )


def plots_root(scratch: Path, judge_id: str) -> Path:
    """Mirror the layout setup/judge_env.sh creates."""
    if judge_id == INCUMBENT:
        return scratch / "rup" / "plots"
    return scratch / "rup" / "judges" / judge_id / "plots"


def load_summary(scratch: Path, judge_id: str, bench: str, models: list[str]) -> pd.DataFrame:
    """Load both cost-summary flavours for one judge into a long frame."""
    rows = []
    root = plots_root(scratch, judge_id)
    for model in models:
        cost_dir = root / bench / model / "cost"
        for axis, fname in [("flops", "cost_summary_metrics.csv"),
                            ("flops_nojudge", "cost_summary_metrics_nojudge.csv")]:
            path = cost_dir / fname
            if not path.exists():
                continue
            df = pd.read_csv(path)
            for _, r in df.iterrows():
                rows.append({
                    "judge": judge_id,
                    "model": model,
                    "attack": r["attack_id"],
                    "axis": axis,
                    # C@0.5 is NaN when the 50% threshold was never crossed; frac_inf marks it.
                    "C@0.5": float("inf") if r.get("C@0.5_frac_inf", 0) == 1.0 else r["C@0.5_mean"],
                    "AE_e3": r["AE_mean"] * 1e3,
                })
    return pd.DataFrame(rows)


def judge_share(scratch: Path, judge_id: str, bench: str, models: list[str]) -> pd.DataFrame:
    """Judge FLOPs as a fraction of total, at the largest lambda of each run."""
    rows = []
    root = plots_root(scratch, judge_id)
    for model in models:
        path = root / bench / model / "cost" / "cost_metrics.csv"
        if not path.exists():
            continue
        d = pd.read_csv(path)
        for attack, g in d.groupby("attack_id"):
            last = g.sort_values("lambda").iloc[-1]
            if not last["mean_total_tflops"]:
                continue
            rows.append({
                "judge": judge_id, "model": model, "attack": attack,
                "judge_frac": last["mean_judge_tflops"] / last["mean_total_tflops"],
            })
    return pd.DataFrame(rows)


def rank_corr(a: pd.Series, b: pd.Series) -> float:
    """Spearman rho between two aligned series, ignoring non-finite pairs."""
    ok = a.notna() & b.notna() & (a != float("inf")) & (b != float("inf"))
    if ok.sum() < 3:
        return float("nan")
    if spearmanr is not None:
        return float(spearmanr(a[ok], b[ok]).statistic)
    return float(a[ok].rank().corr(b[ok].rank()))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--benchmark", default="harmbench",
                   choices=["harmbench", "jailbreakbench"])
    p.add_argument("--judges", nargs="+",
                   default=[INCUMBENT, "olmo3-7b-instruct", "gemma3-4b-it"],
                   help="Judge model_ids. The incumbent must come first — it is the baseline.")
    p.add_argument("--models", nargs="+", default=MAIN_MODELS,
                   help="Target models (default: the nine main-paper targets).")
    p.add_argument("--attacks", nargs="+", default=["gcg", "pair", "jailbroken", "rl"])
    p.add_argument("--scratch", default=None,
                   help="Root containing rup/ (default: auto-detected from $SCRATCH, which "
                        "start_env.sh sets one level deeper than the ambient value).")
    p.add_argument("--out-csv", default=None, help="Optional path to write the joined table.")
    args = p.parse_args()

    scratch = resolve_scratch(args.scratch)
    print(f"Reading from: {scratch}/rup\n")

    frames = [load_summary(scratch, j, args.benchmark, args.models) for j in args.judges]
    long = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    long = long[long.attack.isin(args.attacks)]
    if long.empty:
        raise SystemExit("No cost summaries found. Run run_judge_cost_main.sh first.")

    baseline = args.judges[0]
    present = sorted(long.judge.unique())
    missing = [j for j in args.judges if j not in present]
    if missing:
        print(f"WARNING: no data for judge(s): {', '.join(missing)}\n")

    pd.set_option("display.width", 250)

    for axis in ["flops_nojudge", "flops"]:
        sub = long[long.axis == axis]
        if sub.empty:
            continue
        label = ("target + attacker (judge-size invariant)" if axis == "flops_nojudge"
                 else "target + attacker + judge")
        print("=" * 100)
        print(f"AXIS: {axis}   [{label}]   benchmark={args.benchmark}")
        print("=" * 100)
        for metric in ["C@0.5", "AE_e3"]:
            wide = sub.pivot_table(index=["attack", "model"], columns="judge",
                                   values=metric, aggfunc="first", dropna=False)
            wide = wide.reindex(columns=[j for j in args.judges if j in wide.columns])
            print(f"\n-- {metric} --")
            print(wide.round(2).to_string())

        # Does the ordering of target models survive the judge swap?
        print("\n-- Spearman rho vs. baseline judge "
              f"({baseline}), over target models --")
        rows = []
        for attack in args.attacks:
            s = sub[sub.attack == attack]
            if s.empty:
                continue
            row = {"attack": attack}
            for metric in ["C@0.5", "AE_e3"]:
                w = s.pivot_table(index="model", columns="judge", values=metric,
                                  aggfunc="first", dropna=False)
                if baseline not in w.columns:
                    continue
                for j in args.judges[1:]:
                    if j in w.columns:
                        row[f"{metric}:{j}"] = round(rank_corr(w[baseline], w[j]), 3)
            rows.append(row)
        if rows:
            print(pd.DataFrame(rows).to_string(index=False))
        print()

    # How much of the accounted compute is the instrument rather than the attack?
    shares = pd.concat(
        [judge_share(scratch, j, args.benchmark, args.models) for j in args.judges],
        ignore_index=True)
    if not shares.empty:
        print("=" * 100)
        print("Judge share of total FLOPs (median over target models)")
        print("=" * 100)
        tab = (shares[shares.attack.isin(args.attacks)]
               .pivot_table(index="attack", columns="judge", values="judge_frac",
                            aggfunc="median"))
        tab = tab.reindex(columns=[j for j in args.judges if j in tab.columns])
        print((tab * 100).round(1).to_string())
        print("\n(percent of accounted TFLOPs spent on the judge, at the largest lambda)")

    if args.out_csv:
        long.to_csv(args.out_csv, index=False)
        print(f"\nJoined table written to: {args.out_csv}")


if __name__ == "__main__":
    main()
