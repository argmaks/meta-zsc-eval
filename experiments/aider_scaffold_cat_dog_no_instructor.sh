
#!/bin/bash

# register hydra overrides
# runner_models=("gemini/gemini-2.5-pro" "bedrock/eu.anthropic.claude-sonnet-4-20250514-v1:0")
runner_models=("gemini/gemini-2.5-pro")
feedback_loop_iters=(2 3)
algos=("q_learning_1")
n_runs=3

CMDS=()
SAMPLE_DIRS=()

for model in "${runner_models[@]}"; do
    for feedback_loop_iter in "${feedback_loop_iters[@]}"; do
        for algo in "${algos[@]}"; do
            for run in $(seq 1 $n_runs); do
                SAMPLE_DIR="${PROJECT_DIR}/samples/obl_cat_dog/$(python3 -c 'from datetime import datetime; print(datetime.now().strftime("%Y%m%d_%H%M%S_") + str(datetime.now().microsecond))')"
                mkdir -p $SAMPLE_DIR
                SAMPLE_DIRS+=( $SAMPLE_DIR )
                CMDS+=( "pixi run python run.py --config-name=aider_scaffold_cat_dog_no_instructor runner.model=$model runner.feedback_loop_iters=$feedback_loop_iter algo=$algo workspace_dir=$SAMPLE_DIR 2>&1 | tee $SAMPLE_DIR/output.log" )
            done
        done
    done
done

# Use the reusable tmux job runner
SESSION="obl_cat_dog_no_instructor"
WORKERS=6

# Call the generic tmux job runner script
exec "$(dirname "$0")/tmux_job_runner.sh" "$SESSION" "$WORKERS" "${CMDS[@]}"