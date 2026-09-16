#!/bin/bash
# run_rejudge.sh — paired judge ablation by re-scoring recorded responses offline.
#
# WHY THIS REPLACES `run_judge_ablation.sh` FOR JUDGE COMPARISONS.
# The judge is the early-stopping criterion inside the attack loop, so re-running the sweep with
# a different judge produces different trajectories: the Olmo tree has 4,652 JailbreakBench steps
# against the incumbent's 15,895, and at matched (prompt_id, step) the responses differ. That
# makes the two runs unpaired, so Cohen's kappa is not computable and any difference mixes the
# judge's labelling with a divergent attack path.
#
# Re-judging the INCUMBENT tree's stored responses fixes both problems at once. The trajectory is
# held fixed, so the only thing varying is the judge; the labels are paired, so kappa is
# available; and it costs roughly a tenth of a sweep because only the judge runs.
#
# STAGES
#   gate         diagnose_judge.py per judge against a 22-case labelled control set. HARD GATE:
#                a judge that cannot score this set correctly must not produce a table.
#   validate     re-judge one cell with the INCUMBENT and confirm it reproduces the stored
#                labels. If this does not land near 100% agreement, the replay is wrong
#                (judge_inputs, rubric text, or 4-bit determinism) and nothing downstream holds.
#   judge        re-judge every stored step, per judge (GPU; the expensive stage)
#   materialize  write schema-identical results trees from the sidecars (CPU, fast)
#   eval         metrics.csv + cost CSVs over those trees, main-paper targets only
#   agreement    kappa matrix, flip-by-branch table, refusal FPR, FNR vs severity
#   all          gate -> judge -> materialize -> eval -> agreement
#
# Usage:
#   bash run_rejudge.sh gate
#   bash run_rejudge.sh validate
#   bash run_rejudge.sh all
#   JUDGES="olmo3_7b_instruct_judge" BENCHES="jailbreakbench" bash run_rejudge.sh judge
#
# WHICH AXIS TO READ. The judges differ in size (llama3.1-8b 8.03B, olmo3-7b 7.30B,
# gemma3-4b 3.88B), so on the total `flops` axis part of any cross-judge difference is just the
# judge's own forward pass getting cheaper. Read conclusions on `*_nojudge` (target + attacker),
# which is invariant to judge size; the total axis only quantifies the judge's own share.

# Deliberately NOT -e: a missing model must warn, not abort (same reason as
# run_judge_cost_main.sh). Not -u either — setup/start_env.sh reads $OFFLINE_MODE with no default.
set -o pipefail

: "${OFFLINE_MODE:=0}"
export OFFLINE_MODE

source setup/start_env.sh

STAGE="${1:-all}"

# Judge CONFIG names (configs/models/<name>.yaml). The incumbent is included: re-judging with it
# is the validation run, and its own row is needed for the paired comparison.
JUDGES="${JUDGES:-llama3.1_8b_instruct_judge gemma3_4b_it_judge flow_judge_v01}"
BENCHES="${BENCHES:-jailbreakbench harmbench}"

# Source of truth: the incumbent tree, whose trajectories were not truncated by a broken judge.
#
# CAUTION on $SCRATCH. setup/start_env.sh picks it from the hostname — `klogin*` gets
# /home/$USER/scratch/$USER, everything else gets /scratch/$USER. Compute nodes here are `kn*`,
# not `klogin*`, so inside a batch job $SCRATCH resolves to /scratch/$USER, where no rup tree
# exists. The other scripts survive that only because they expand $SCRATCH on the LOGIN node at
# submit time. Run this script on the login node, or pass SRC_ROOT explicitly. The existence check
# below turns a silent "0 cells found" into a loud failure either way.
SRC_ROOT="${SRC_ROOT:-$SCRATCH/rup}"
REJUDGE_ROOT="${REJUDGE_ROOT:-$SCRATCH/rup/rejudge}"

if [ ! -d "$SRC_ROOT" ]; then
    echo "ERROR: SRC_ROOT does not exist: $SRC_ROOT" >&2
    echo "  \$SCRATCH is currently '$SCRATCH' (hostname: $(hostname))." >&2
    echo "  If this is a compute node, \$SCRATCH is not what the login node uses — see the" >&2
    echo "  comment above. Re-run on the login node, or set SRC_ROOT explicitly:" >&2
    echo "    SRC_ROOT=/home/\$USER/scratch/\$USER/rup bash run_rejudge.sh $STAGE" >&2
    exit 1
fi

# The nine targets of Table 1. Excludes the gemma3-* / olmo2-* ladders added later.
MAIN_MODELS="${MAIN_MODELS:-tulu3-8b-base tulu3-8b-sft tulu3-8b-dpo tulu3-8b-rlvr \
qwen2.5-0.5b-instruct qwen2.5-3b-instruct qwen2.5-7b-instruct qwen3-4b qwen3-4b-saferl}"

VALIDATE_CELL="${VALIDATE_CELL:-jailbreakbench/tulu3-8b-dpo}"

judge_id_of() {   # config name -> model_id, read from the YAML so keys cannot drift
    python -c "import yaml,sys; print(yaml.safe_load(open('configs/models/$1.yaml'))['model_id'])"
}

skipped=0
done_n=0

case "$STAGE" in

gate|all)
    echo "=================== STAGE: gate ==================="
    gate_failed=0
    mkdir -p "$REJUDGE_ROOT/diagnostics"
    for judge in $JUDGES; do
        echo "--- $judge"
        python scripts/diagnose_judge.py --judge "$judge" \
            --min-accuracy "${MIN_ACCURACY:-1.0}" \
            --json-out "$REJUDGE_ROOT/diagnostics/diag_${judge}.json" \
            || { echo "GATE FAIL: $judge" >&2; gate_failed=1; }
    done
    if [ "$gate_failed" -ne 0 ]; then
        echo >&2
        echo "GATE FAILED — a judge that cannot score the labelled control set must not" >&2
        echo "produce a table. Fix it, or drop it from JUDGES, before continuing." >&2
        exit 1
    fi
    echo "All judges passed the control-set gate."
    [ "$STAGE" = "gate" ] && exit 0
    ;;&

validate)
    echo "=================== STAGE: validate ==================="
    echo "Re-judging $VALIDATE_CELL with the INCUMBENT judge."
    echo "The fixed parser must reproduce the stored labels; anything below ~99% means the"
    echo "replay itself is wrong and no re-judged table can be trusted."
    python scripts/rejudge_offline.py judge \
        --results-dir "$SRC_ROOT/$VALIDATE_CELL" \
        --judge-model llama3.1_8b_instruct_judge || exit 1

    JID=$(judge_id_of llama3.1_8b_instruct_judge)
    python - "$SRC_ROOT/$VALIDATE_CELL" "$JID" <<'PY'
import json, re, sys
from pathlib import Path
root, judge_id = Path(sys.argv[1]), sys.argv[2]
name = f"rejudge__{re.sub(r'[^a-zA-Z0-9]+','_',judge_id).strip('_')}.jsonl"
agree = total = unparsed = 0
by_attack, disagreements = {}, []
for f in sorted(root.rglob(name)):
    for line in f.open():
        row = json.loads(line)
        a = row["attack_id"]
        st = by_attack.setdefault(a, [0, 0])
        for s in row["steps"]:
            total += 1; st[1] += 1
            if not s["parsed"]:
                unparsed += 1
            if s["judgment"] == s["orig_judgment"]:
                agree += 1; st[0] += 1
            else:
                disagreements.append((a, row["prompt_id"], s["step"],
                                      s["orig_judgment"], s["judgment"], s["branch"]))
print(f"\nself-consistency: {agree}/{total} = {agree/max(total,1):.4f}   unparsed={unparsed}")
for a, (ok, n) in sorted(by_attack.items()):
    flag = "" if ok/max(n,1) >= 0.99 else "   <-- BELOW 99%"
    print(f"  {a:12s} {ok}/{n} = {ok/max(n,1):.4f}{flag}")
if disagreements:
    print(f"\n{len(disagreements)} disagreement(s); every one must be explainable by branch:")
    from collections import Counter
    print("  branches:", dict(Counter(d[5] for d in disagreements)))
    for d in disagreements[:20]:
        print(f"  {d[0]}/{d[1]} step {d[2]}: stored={d[3]} new={d[4]} branch={d[5]}")
ok = agree/max(total,1) >= 0.99
print("\nVALIDATE:", "PASS" if ok else "FAIL — do not build tables from a re-judge")
sys.exit(0 if ok else 1)
PY
    exit $?
    ;;

esac

case "$STAGE" in

judge|all)
    echo "=================== STAGE: judge ==================="
    for judge in $JUDGES; do
        for bench in $BENCHES; do
            for model in $MAIN_MODELS; do
                rdir="$SRC_ROOT/$bench/$model"
                if [ ! -d "$rdir" ]; then
                    echo "SKIP  $judge / $bench / $model  (no results dir)"
                    skipped=$((skipped + 1)); continue
                fi
                echo "JUDGE $judge / $bench / $model"
                python scripts/rejudge_offline.py judge \
                    --results-dir "$rdir" --judge-model "$judge" \
                    || { echo "WARN  rejudge failed: $judge / $bench / $model" >&2
                         skipped=$((skipped + 1)); continue; }
                done_n=$((done_n + 1))
            done
        done
    done
    [ "$STAGE" = "judge" ] && { echo "Done: $done_n cells, $skipped skipped."; exit 0; }
    ;;&

materialize|all)
    echo "=================== STAGE: materialize ==================="
    for judge in $JUDGES; do
        JID=$(judge_id_of "$judge") || continue
        for bench in $BENCHES; do
            for model in $MAIN_MODELS; do
                rdir="$SRC_ROOT/$bench/$model"
                [ -d "$rdir" ] || continue
                echo "MATER $JID / $bench / $model"
                python scripts/rejudge_offline.py materialize \
                    --results-dir "$rdir" --judge-model "$judge" \
                    --out-root "$REJUDGE_ROOT/$JID/$bench/$model" \
                    --unparsed-policy "${UNPARSED_POLICY:-safe}" \
                    || { echo "WARN  materialize failed: $JID / $bench / $model" >&2
                         skipped=$((skipped + 1)); }
            done
        done
    done
    [ "$STAGE" = "materialize" ] && exit 0
    ;;&

eval|all)
    echo "=================== STAGE: eval ==================="
    EVAL="python scripts/run_evaluation.py --experiment configs/experiments/base.yaml \
--format csv --print-table"
    for judge in $JUDGES; do
        JID=$(judge_id_of "$judge") || continue
        for bench in $BENCHES; do
            for model in $MAIN_MODELS; do
                rdir="$REJUDGE_ROOT/$JID/$bench/$model"
                odir="$REJUDGE_ROOT/$JID/plots/$bench/$model"
                if [ ! -d "$rdir" ]; then
                    echo "SKIP  $JID / $bench / $model  (not materialized)"
                    skipped=$((skipped + 1)); continue
                fi
                mkdir -p "$odir/cost"
                echo "EVAL  $JID / $bench / $model"
                $EVAL --results-dir "$rdir" --output "$odir/metrics.csv" \
                    | tee "$rdir/summary.txt" \
                    || { echo "WARN  eval failed: $JID / $bench / $model" >&2
                         skipped=$((skipped + 1)); continue; }

                echo "COST  $JID / $bench / $model"
                python scripts/compute_attack_costs.py \
                    --pricing-config configs/pricing.yaml \
                    --judge-model "$JID" \
                    --results-dir "$rdir" \
                    --metrics-csv "$odir/metrics.csv" \
                    --output      "$odir/cost/cost_metrics.csv" \
                    || { echo "WARN  cost failed: $JID / $bench / $model" >&2
                         skipped=$((skipped + 1)); }
            done
        done
    done
    [ "$STAGE" = "eval" ] && exit 0
    ;;&

agreement|all)
    echo "=================== STAGE: agreement ==================="
    JUDGE_IDS=""
    for judge in $JUDGES; do
        JUDGE_IDS="$JUDGE_IDS $(judge_id_of "$judge")"
    done
    for bench in $BENCHES; do
        echo "--- $bench"
        python scripts/judge_agreement.py \
            --results-dir "$SRC_ROOT/$bench" \
            --judges $JUDGE_IDS \
            --json-out "$REJUDGE_ROOT/agreement_$bench.json" \
            || echo "WARN  agreement failed for $bench" >&2
    done
    ;;

*)
    echo "ERROR: unknown stage '$STAGE'" >&2
    echo "Expected: gate | validate | judge | materialize | eval | agreement | all" >&2
    exit 1
    ;;
esac

echo
echo "Done: $done_n cells processed, $skipped skipped."
echo
echo "Read the results on the *_nojudge axis:"
echo "  \$REJUDGE_ROOT/<judge_id>/plots/<bench>/<model>/cost/cost_summary_metrics_nojudge.csv"
echo "Censoring (how much of the risk curve is a lower bound) is in each tree's"
echo "  rejudge_manifest.json — quote risk as an envelope if censored_frac is non-trivial."
