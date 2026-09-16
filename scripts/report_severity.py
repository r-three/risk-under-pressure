#!/usr/bin/env python3
"""
report_severity.py — roll the per-model severity CSVs up into the paper's study tables.

score_severity.py writes one severity_metrics.csv per target model. This script collects
them and answers the comparison questions directly:

  * model size        — does a bigger Qwen2.5 give away *worse* content, not just more often?
  * training stage    — Tulu3 base -> SFT -> DPO -> RLVR: which stage moves severity?
  * safety alignment  — Qwen3-4B vs Qwen3-4B-SafeRL: what does safety RL actually buy?
  * attack            — how much extra severity the adaptive RL (GRPO) attacker extracts
                        over the static attacks, per target.

Everything is reported at the maximum pressure level present, which is the end of the
budget: the point where the query budget is exhausted or every trial has early-stopped.
Per-λ rows are carried through to the CSV so the trajectory is still available.

Where a seed-aggregated severity_summary.csv exists it is preferred over
severity_metrics.csv, so multi-seed runs report the across-seed mean.

Usage:
    # Scan a whole plots tree:
    python scripts/report_severity.py --root $SCRATCH/rup/plots

    # Restrict to one benchmark, write elsewhere:
    python scripts/report_severity.py \\
        --root $SCRATCH/rup/plots/harmbench \\
        --output $SCRATCH/rup/plots/harmbench/severity_report.csv

    # Explicit inputs instead of a scan:
    python scripts/report_severity.py --inputs a/severity_metrics.csv b/severity_metrics.csv
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rup.judges.severity_judge import SEVERITY_LABELS, SEVERITY_MAX
from rup.utils.logging import get_logger

logger = get_logger("report_severity")

# Study groupings. `order` is both the membership list and the axis order; the first entry
# is the reference the Δ column is measured against.
STUDIES: dict[str, dict] = {
    "model_size": {
        "title": "MODEL SIZE — Qwen2.5-Instruct",
        "order": ["qwen2.5-0.5b-instruct", "qwen2.5-3b-instruct", "qwen2.5-7b-instruct"],
    },
    "model_size_gemma3": {
        "title": "MODEL SIZE — Gemma 3 Instruction-Tuned (2nd family)",
        "order": ["gemma3-270m-it", "gemma3-1b-it", "gemma3-4b-it"],
    },
    "training_stage": {
        "title": "TRAINING STAGE — Tulu3 8B (base -> SFT -> DPO -> RLVR)",
        "order": ["tulu3-8b-base", "tulu3-8b-sft", "tulu3-8b-dpo", "tulu3-8b-rlvr"],
    },
    "training_stage_olmo2": {
        # Five rungs: allenai's `-Instruct` is a second RLVR round stacked on `-RLVR1`, so
        # the last segment of this table is the marginal effect of one further RLVR pass —
        # something the four-rung Tulu3 ladder cannot show.
        "title": "TRAINING STAGE — OLMo 2 1B (base -> SFT -> DPO -> RLVR1 -> RLVR2)",
        "order": ["olmo2-1b-base", "olmo2-1b-sft", "olmo2-1b-dpo",
                  "olmo2-1b-rlvr1", "olmo2-1b-instruct"],
    },
    "training_stage_tulu2": {
        "title": "TRAINING STAGE — Tulu2 7B (base -> SFT -> DPO)",
        "order": ["tulu2-7b-base", "tulu2-7b-sft", "tulu2-7b-dpo"],
    },
    "safety_alignment": {
        "title": "SAFETY RL — Qwen3-4B vs Qwen3-4B-SafeRL",
        "order": ["qwen3-4b", "qwen3-4b-saferl"],
    },
}

MODEL_DISPLAY = {
    "qwen2.5-0.5b-instruct": "Qwen2.5-0.5B",
    "qwen2.5-3b-instruct":   "Qwen2.5-3B",
    "qwen2.5-7b-instruct":   "Qwen2.5-7B",
    "gemma3-270m-it":        "Gemma3-270M",
    "gemma3-1b-it":          "Gemma3-1B",
    "gemma3-4b-it":          "Gemma3-4B",
    "tulu3-8b-base":         "Tulu3-Base",
    "tulu3-8b-sft":          "Tulu3-SFT",
    "tulu3-8b-dpo":          "Tulu3-DPO",
    "tulu3-8b-rlvr":         "Tulu3-RLVR",
    "olmo2-1b-base":         "OLMo2-Base",
    "olmo2-1b-sft":          "OLMo2-SFT",
    "olmo2-1b-dpo":          "OLMo2-DPO",
    "olmo2-1b-rlvr1":        "OLMo2-RLVR1",
    "olmo2-1b-instruct":     "OLMo2-RLVR2",
    "tulu2-7b-base":         "Tulu2-Base",
    "tulu2-7b-sft":          "Tulu2-SFT",
    "tulu2-7b-dpo":          "Tulu2-DPO",
    "qwen3-4b":              "Qwen3-4B",
    "qwen3-4b-saferl":       "Qwen3-4B-SafeRL",
    "qwen3-8b":              "Qwen3-8B",
}

ATTACK_DISPLAY = {
    "gcg": "GCG", "pair": "PAIR", "jailbroken": "JailBroken",
    "jailbroken-v1": "JailBroken-v1", "rl": "RL (GRPO)",
}

MODEL_TO_STUDY = {m: s for s, meta in STUDIES.items() for m in meta["order"]}
_SEED_RE = re.compile(r"_seed\d+$")


def _model_label(m: str) -> str:
    return MODEL_DISPLAY.get(m, m)


def _attack_label(a: str) -> str:
    if a in ATTACK_DISPLAY:
        return ATTACK_DISPLAY[a]
    base, sep, attacker = a.partition("__")
    if sep:
        return f"{ATTACK_DISPLAY.get(base, base)} — {attacker}"
    return a


def _arm_suffix(path: Path) -> str:
    """The rubric tag carried by a severity CSV name ('' for the uplift arm, '_detail', ...).

    Recovered from the filename so the per-category companion is read from the same arm as
    the metrics file it accompanies, whether that path came from --root or --inputs.
    """
    stem = path.stem
    for prefix in ("severity_metrics_by_category", "severity_by_category_summary",
                   "severity_metrics", "severity_summary"):
        if stem.startswith(prefix):
            return stem[len(prefix):]
    return ""


def discover_inputs(root: Path, metrics_name: str = "severity_metrics.csv") -> list[Path]:
    """Find every severity CSV under `root`, preferring the seed-aggregated one per dir.

    `metrics_name` selects the rubric arm: run_severity_scoring.sh suffixes the CSVs it
    writes with the rubric (severity_metrics_detail.csv), so a report that globbed the bare
    name would silently table the *other* arm's numbers.
    """
    stem = metrics_name[:-len(".csv")] if metrics_name.endswith(".csv") else metrics_name
    suffix = stem[len("severity_metrics"):] if stem.startswith("severity_metrics") else ""
    dirs = {p.parent for p in root.rglob(metrics_name)}
    out = []
    for d in sorted(dirs):
        summary = d / f"severity_summary{suffix}.csv"
        out.append(summary if summary.exists() else d / metrics_name)
    return out


def infer_benchmark(path: Path) -> str:
    """Read the benchmark out of the path (…/<benchmark>/<model>/severity_*.csv)."""
    for part in path.parts[::-1]:
        low = part.lower()
        if low in ("harmbench", "jailbreakbench"):
            return low
    return "unknown"


def load_frame(paths: list[Path]) -> pd.DataFrame:
    """Concatenate the severity CSVs into one long frame with benchmark/study columns."""
    frames = []
    for p in paths:
        if not p.exists():
            logger.warning(f"Missing: {p}")
            continue
        df = pd.read_csv(p)
        if df.empty:
            logger.warning(f"Empty: {p}")
            continue
        df["benchmark"] = infer_benchmark(p)
        df["source_csv"] = str(p)
        # Collapse any _seedN suffix that survived (single-seed runs never get a summary CSV).
        df["model_id"] = df["model_id"].astype(str).str.replace(_SEED_RE, "", regex=True)
        frames.append(df)

    if not frames:
        logger.error("No severity CSVs could be read.")
        sys.exit(1)

    df = pd.concat(frames, ignore_index=True)
    # A directory scanned twice (metrics + summary) or multi-seed rows that were never
    # aggregated would otherwise double-count; average them.
    num_cols = df.select_dtypes("number").columns.tolist()
    keys = ["benchmark", "model_id", "attack_id", "lambda"]
    df = df.groupby(keys, as_index=False)[num_cols].mean()
    df["study"] = df["model_id"].map(MODEL_TO_STUDY).fillna("other")
    return df


def endpoints(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (benchmark, model, attack) at the largest λ reported.

    That is the end of the budget: every trial has either spent all λ_max queries or
    early-stopped at its first success, so this is the final severity attributable to
    the full attack.
    """
    idx = df.groupby(["benchmark", "model_id", "attack_id"])["lambda"].idxmax()
    return df.loc[idx].reset_index(drop=True)


def _fmt(v, fmt=".2f", width=0) -> str:
    try:
        s = format(float(v), fmt)
    except (TypeError, ValueError):
        s = "n/a"
    return s.rjust(width) if width else s


def study_table(end: pd.DataFrame, study: str, benchmark: str) -> str | None:
    """Severity per model × attack for one study on one benchmark, with Δ vs the reference."""
    meta = STUDIES[study]
    order = [m for m in meta["order"]
             if not end[(end.benchmark == benchmark) & (end.model_id == m)].empty]
    if len(order) < 1:
        return None

    sub = end[(end.benchmark == benchmark) & (end.model_id.isin(order))]
    attacks = sorted(sub.attack_id.unique())
    ref = order[0]

    lines = [
        f"{meta['title']}   [{benchmark}]",
        f"reference for Δ: {_model_label(ref)}",
        "",
        f"{'attack':<20} {'model':<18} {'λ':>3} {'sev':>6} {'Δsev':>7} "
        f"{'sev|succ':>9} {'≥thr':>6} {'ASR':>6} {'AUSC':>7} {'N':>5}",
        "─" * 96,
    ]
    for attack in attacks:
        rows = sub[sub.attack_id == attack].set_index("model_id")
        base = rows.loc[ref, "mean_peak_severity"] if ref in rows.index else None
        for i, model in enumerate(order):
            if model not in rows.index:
                continue
            r = rows.loc[model]
            delta = "  ref" if (base is None or model == ref) else _fmt(
                r["mean_peak_severity"] - base, "+.2f")
            lines.append(
                f"{_attack_label(attack) if i == 0 else '':<20} "
                f"{_model_label(model):<18} {int(r['lambda']):>3} "
                f"{_fmt(r['mean_peak_severity'], '.2f', 6)} {delta:>7} "
                f"{_fmt(r['mean_severity_given_success'], '.2f', 9)} "
                f"{_fmt(r['frac_severe'], '.2f', 6)} {_fmt(r['asr'], '.2f', 6)} "
                f"{_fmt(r['au_sev_c'], '.2f', 7)} {int(float(r['n_prompts'])):>5}"
            )
        lines.append("")
    return "\n".join(lines)


def attack_table(end: pd.DataFrame, benchmark: str) -> str | None:
    """Severity per attack per target, plus the RL attacker's lift over the best static one."""
    sub = end[end.benchmark == benchmark]
    if sub.empty:
        return None

    attacks = sorted(sub.attack_id.unique())
    static = [a for a in attacks if not a.split("__")[0].startswith("rl")]
    rl = [a for a in attacks if a.split("__")[0].startswith("rl")]

    header = (f"{'model':<18} " + "".join(f"{_attack_label(a)[:11]:>12}" for a in attacks)
              + f"{'RL lift':>10}")
    lines = [
        f"ATTACK EFFECT — mean peak severity (0-{SEVERITY_MAX}) at end of budget   [{benchmark}]",
        "",
        header,
        "─" * len(header),
    ]
    models = sorted(sub.model_id.unique(),
                    key=lambda m: (list(MODEL_TO_STUDY).index(m) if m in MODEL_TO_STUDY else 999, m))
    for model in models:
        rows = sub[sub.model_id == model].set_index("attack_id")
        cells = "".join(
            _fmt(rows.loc[a, "mean_peak_severity"], ".2f", 12) if a in rows.index else f"{'—':>12}"
            for a in attacks
        )
        best_static = max((rows.loc[a, "mean_peak_severity"] for a in static if a in rows.index),
                          default=None)
        best_rl = max((rows.loc[a, "mean_peak_severity"] for a in rl if a in rows.index),
                      default=None)
        lift = _fmt(best_rl - best_static, "+.2f", 10) if (best_rl is not None and best_static is not None) else f"{'—':>10}"
        lines.append(f"{_model_label(model):<18}{cells}{lift}")
    lines.append("")
    lines.append("RL lift = best RL (GRPO) attack minus best static attack, same target.")
    return "\n".join(lines)


def category_table(cat_df: pd.DataFrame, benchmark: str, top_n: int) -> str | None:
    """Harm categories ranked by mean severity across all targets and attacks."""
    sub = cat_df[cat_df.benchmark == benchmark]
    if sub.empty:
        return None
    end_idx = sub.groupby(["model_id", "attack_id", "category"])["lambda"].idxmax()
    end = sub.loc[end_idx]
    agg = (end.groupby("category")
              .agg(sev=("mean_peak_severity", "mean"),
                   asr=("asr", "mean"),
                   frac=("frac_severe", "mean"))
              .sort_values("sev", ascending=False).head(top_n))

    lines = [f"HARM CATEGORY — mean peak severity across all targets/attacks   [{benchmark}]", "",
             f"{'category':<34} {'sev':>6} {'≥thr':>6} {'ASR':>6}", "─" * 55]
    for cat, r in agg.iterrows():
        lines.append(f"{str(cat)[:34]:<34} {_fmt(r['sev'], '.2f', 6)} "
                     f"{_fmt(r['frac'], '.2f', 6)} {_fmt(r['asr'], '.2f', 6)}")
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate severity CSVs into study tables")
    p.add_argument("--root", default=None,
                   help="Directory to scan recursively for severity_metrics.csv")
    p.add_argument("--metrics-name", default="severity_metrics.csv",
                   help="Which per-model CSV to collect under --root. Selects the rubric arm: "
                        "severity_metrics.csv is the uplift arm, severity_metrics_detail.csv "
                        "the detail arm (default: severity_metrics.csv)")
    p.add_argument("--inputs", nargs="+", default=None,
                   help="Explicit severity CSV paths (alternative to --root)")
    p.add_argument("--output", default=None,
                   help="Report CSV path (default: <root>/severity_report.csv)")
    p.add_argument("--benchmarks", nargs="+", default=None,
                   help="Restrict to these benchmarks (default: all found)")
    p.add_argument("--top-categories", type=int, default=12,
                   help="Rows in the harm-category table (default: 12)")
    p.add_argument("--no-categories", action="store_true",
                   help="Skip the per-category table even if the CSVs exist")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.root and not args.inputs:
        logger.error("Pass --root or --inputs.")
        sys.exit(1)

    if args.inputs:
        paths = [Path(p) for p in args.inputs]
    else:
        paths = discover_inputs(Path(args.root), args.metrics_name)
    if not paths:
        logger.error(f"No {args.metrics_name} found under {args.root}")
        sys.exit(1)
    logger.info(f"Reading {len(paths)} severity CSVs")

    df = load_frame(paths)
    end = endpoints(df)

    benchmarks = args.benchmarks or sorted(df.benchmark.unique())

    cat_df = None
    if not args.no_categories:
        cat_paths = [p.parent / f"severity_metrics_by_category{_arm_suffix(p)}.csv" for p in paths]
        cat_paths = [p for p in dict.fromkeys(cat_paths) if p.exists()]
        if cat_paths:
            frames = []
            for p in cat_paths:
                d = pd.read_csv(p)
                if d.empty:
                    continue
                d["benchmark"] = infer_benchmark(p)
                d["model_id"] = d["model_id"].astype(str).str.replace(_SEED_RE, "", regex=True)
                frames.append(d)
            if frames:
                cat_df = pd.concat(frames, ignore_index=True)

    print()
    print("=" * 96)
    print(f"SEVERITY REPORT — graded harm on a 0-{SEVERITY_MAX} scale "
          f"({'; '.join(f'{k}={v}' for k, v in SEVERITY_LABELS.items())})")
    print("Reported at the end of the query budget: every trial has either spent λ_max")
    print("queries or early-stopped at its first successful jailbreak.")
    print("=" * 96)

    for benchmark in benchmarks:
        for study in STUDIES:
            table = study_table(end, study, benchmark)
            if table:
                print()
                print(table)
        table = attack_table(end, benchmark)
        if table:
            print()
            print(table)
        if cat_df is not None:
            table = category_table(cat_df, benchmark, args.top_categories)
            if table:
                print()
                print(table)

    out_cols = [
        "benchmark", "study", "model_id", "attack_id", "lambda",
        "mean_peak_severity", "peak_severity_lower", "peak_severity_upper",
        "mean_peak_severity_norm", "mean_step_severity", "mean_severity_given_success",
        "frac_severe", "asr", "au_sev_c", "delta_sev", "mean_queries_used",
        "unparsed_rate", "n_prompts", "n_success",
    ]
    out_cols = [c for c in out_cols if c in df.columns]

    output = Path(args.output) if args.output else Path(args.root or ".") / "severity_report.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    df.sort_values(["benchmark", "study", "model_id", "attack_id", "lambda"])[out_cols].to_csv(
        output, index=False, na_rep="")
    print(f"\nFull severity report (all λ) written to: {output}")

    end_output = output.parent / f"{output.stem}_endpoints.csv"
    end.sort_values(["benchmark", "study", "model_id", "attack_id"])[out_cols].to_csv(
        end_output, index=False, na_rep="")
    print(f"End-of-budget rows written to:           {end_output}")


if __name__ == "__main__":
    main()
