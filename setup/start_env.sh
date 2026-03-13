if [[ $(hostname) == "trig-login01" ]]; then
    module load StdEnv/2023
fi
module load cuda/12.6
module load gcc arrow/19.0.1 python/3.11

source .venv/bin/activate

export HF_HOME=~/.cache/huggingface
export HF_DATASETS_CACHE=~/.cache/huggingface/datasets

# Load API keys from .env if present
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

# HuggingFace token: .env takes priority, fall back to ~/hf_token.txt
if [ -f ~/hf_token.txt ]; then
    python -c "from huggingface_hub import login; login(token='$(cat ~/hf_token.txt)')"
fi

# If OFFLINE_MODE is set to 1, disable network access
if [[ $OFFLINE_MODE == 1 ]]; then
    echo "Running in offline mode"
    export HF_DATASETS_OFFLINE=1
    export HF_HUB_OFFLINE=1
    export WANDB_MODE=offline
fi


# Check if job should be skipped (already running/pending or completed in past 2 days)
function should_skip_job() {
    local job_name="$1"

    # Check if job is currently running or pending (same user)
    if squeue --name="$job_name" --user="$USER" --noheader 2>/dev/null | grep -q .; then
        echo "SKIP: Job '$job_name' is already running or pending."
        return 0
    fi

    # Check if job completed successfully in the past 2 days (same user)
    local two_days_ago=$(date -d '2 days ago' +%Y-%m-%d 2>/dev/null || date -v-2d +%Y-%m-%d)
    if sacct --name="$job_name" --user="$USER" --starttime="$two_days_ago" --state=COMPLETED --noheader 2>/dev/null | grep -q .; then
        echo "SKIP: Job '$job_name' completed successfully in the past 2 days."
        return 0
    fi

    return 1
}

if [[ $(hostname) == klogin* ]]; then
    # define job submission function (killarney L40S)
    function submit() {
        local job_name="$1"
        local command="$2"
        should_skip_job "$job_name" && return 0
        mkdir -p logs
        sbatch --job-name="$job_name" --output="logs/%j_$job_name.out" --error="logs/%j_$job_name.out" setup/submit_killarney.sbatch "$command"
    }
else
    echo "Unknown hostname: $(hostname) - cannot define submit function"
fi
