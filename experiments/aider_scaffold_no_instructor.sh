
#!/bin/bash

# register hydra overrides
# runner_models=("gemini/gemini-2.5-pro" "openai/o3" "bedrock/eu.anthropic.claude-sonnet-4-20250514-v1:0")
runner_models=("gemini/gemini-2.5-pro")
feedback_loop_iters=(2 3)
algos=("q_learning_1" "q_learning_2" "q_learning_3")
n_runs=3

CMDS=()
SAMPLE_DIRS=()

for model in "${runner_models[@]}"; do
    for feedback_loop_iter in "${feedback_loop_iters[@]}"; do
        for algo in "${algos[@]}"; do
            for run in $(seq 1 $n_runs); do
                SAMPLE_DIR="${PROJECT_DIR}/samples/few_shot_aider_coding/$(python3 -c 'from datetime import datetime; print(datetime.now().strftime("%Y%m%d_%H%M%S_") + str(datetime.now().microsecond))')"
                mkdir -p $SAMPLE_DIR
                SAMPLE_DIRS+=( $SAMPLE_DIR )
                CMDS+=( "pixi run python run.py --config-name=aider_scaffold_no_instructor runner.model=$model runner.feedback_loop_iters=$feedback_loop_iter algo=$algo workspace_dir=$SAMPLE_DIR 2>&1 | tee $SAMPLE_DIR/output.log" )
            done
        done
    done
done



SESSION="aider_scaffold_no_instructor"      # tmux session name
WORKERS=18             # number of windows (= max parallel runs)

exec "$(dirname "$0")/tmux_job_runner.sh" "$SESSION" "$WORKERS" "${CMDS[@]}"