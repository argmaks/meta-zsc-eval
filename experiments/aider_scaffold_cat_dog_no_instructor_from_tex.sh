
#!/bin/bash

# register hydra overrides
# runner_models=("gemini/gemini-2.5-pro" "openai/o3" "bedrock/eu.anthropic.claude-sonnet-4-20250514-v1:0")
runner_models=("gemini/gemini-2.5-pro")
feedback_loop_iters=(0 1)
algos=("q_learning_1" "q_learning_3")
# Define different sets of tex files to loop over
tex_file_sets=(
    "obl/abstract.tex,obl/background.tex,obl/method.tex"
)
n_runs=5

CMDS=()
SAMPLE_DIRS=()

for model in "${runner_models[@]}"; do
    for feedback_loop_iter in "${feedback_loop_iters[@]}"; do
        for algo in "${algos[@]}"; do
            for tex_files in "${tex_file_sets[@]}"; do
                for run in $(seq 1 $n_runs); do
                    SAMPLE_DIR="${PROJECT_DIR}/samples/obl_cat_dog/$(python3 -c 'from datetime import datetime; print(datetime.now().strftime("%Y%m%d_%H%M%S_") + str(datetime.now().microsecond))')"
                    mkdir -p $SAMPLE_DIR
                    SAMPLE_DIRS+=( $SAMPLE_DIR )
                    CMDS+=( "pixi run python run.py --config-name=aider_scaffold_cat_dog_no_instructor_from_tex runner.model=$model runner.feedback_loop_iters=$feedback_loop_iter algo=$algo prompt.contents.tex_files=\[$tex_files\] workspace_dir=$SAMPLE_DIR 2>&1 | tee $SAMPLE_DIR/output.log" )
                done
            done
        done
    done
done



# Use the reusable tmux job runner
SESSION="obl_cat_dog_no_instructor_from_tex"
WORKERS=20

# Call the generic tmux job runner script
exec "$(dirname "$0")/tmux_job_runner.sh" "$SESSION" "$WORKERS" "${CMDS[@]}"