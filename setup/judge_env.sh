# Judge selection for the run_*.sh scripts. Source AFTER setup/start_env.sh.
#
# Set JUDGE to a model config name under configs/models/ (without .yaml). It defaults
# to the judge in configs/experiments/base.yaml, and in that default case every path
# and job name below is byte-identical to what the scripts used before judges became
# selectable — existing results stay where they are.
#
#   bash run_HB_experiments.sh                            # llama3.1-8b judge (default)
#   JUDGE=gemma3_4b_it_judge bash run_HB_experiments.sh   # gemma3-4b judge
#
# Exports:
#   JUDGE      config name -> run_inference.py --judge-model
#   JUDGE_ID   model_id from that config -> compute_attack_costs.py --judge-model.
#              This is the key cost_mapper uses to look up the judge's params_b (FLOP
#              axis) and its $/1M-token rate (dollar axis), so it must be the model_id,
#              not the config filename.
#   RUN_ROOT   inference output root, one tree per judge so runs never overwrite
#   PLOT_ROOT  metrics/plots root, mirroring RUN_ROOT
#   JUDGE_TAG  suffix for SLURM job names so per-judge jobs don't collide (and so
#              should_skip_job doesn't mistake another judge's run for this one)

DEFAULT_JUDGE="llama3.1_8b_instruct_judge"
JUDGE="${JUDGE:-$DEFAULT_JUDGE}"

_judge_cfg="configs/models/${JUDGE}.yaml"
if [ ! -f "$_judge_cfg" ]; then
    echo "ERROR: judge config not found: $_judge_cfg" >&2
    echo -n "Available judges: " >&2
    ls configs/models/*_judge.yaml 2>/dev/null | xargs -n1 basename | sed 's/\.yaml$//' | tr '\n' ' ' >&2
    echo >&2
    return 1 2>/dev/null || exit 1
fi

# Read model_id from the config rather than hardcoding a second copy of it here,
# so the cost/pricing key can never drift from the config the run actually loaded.
JUDGE_ID=$(python -c "import yaml; print(yaml.safe_load(open('$_judge_cfg'))['model_id'])")
if [ -z "$JUDGE_ID" ]; then
    echo "ERROR: could not read model_id from $_judge_cfg" >&2
    return 1 2>/dev/null || exit 1
fi

if [ "$JUDGE" = "$DEFAULT_JUDGE" ]; then
    RUN_ROOT="$SCRATCH/rup"
    JUDGE_TAG=""
else
    RUN_ROOT="$SCRATCH/rup/judges/$JUDGE_ID"
    JUDGE_TAG="_j_$(echo "$JUDGE_ID" | tr '.-' '__')"
fi
PLOT_ROOT="$RUN_ROOT/plots"

export JUDGE JUDGE_ID RUN_ROOT PLOT_ROOT JUDGE_TAG

echo "Judge:        $JUDGE (model_id=$JUDGE_ID)"
echo "Results root: $RUN_ROOT"
echo "Plots root:   $PLOT_ROOT"
