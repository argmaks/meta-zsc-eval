#!/bin/bash

# Reusable tmux job runner script
# Usage: ./tmux_job_runner.sh SESSION_NAME WORKERS_COUNT "command1" "command2" "command3" ...

if [ $# -lt 3 ]; then
    echo "Usage: $0 SESSION_NAME WORKERS_COUNT command1 [command2] [command3] ..."
    echo "Example: $0 my_experiment 4 'echo job1' 'echo job2' 'echo job3'"
    exit 1
fi

SESSION="$1"
WORKERS="$2"
shift 2  # Remove first two arguments, leaving only commands
CMDS=("$@")  # Remaining arguments are the commands

echo "Starting tmux job runner:"
echo "  Session: $SESSION"
echo "  Workers: $WORKERS"
echo "  Commands: ${#CMDS[@]}"

# Kill existing session if it exists
# tmux kill-session -t "$SESSION" 2>/dev/null || true

# —— SETUP TMUX —— 
tmux new-session -d -s "$SESSION"          # start detached session with first window

# Create additional windows 
ACTUAL_WORKERS=1  # start with 1 (the initial window)
for ((i=1; i<WORKERS; i++)); do
  tmux new-window -t "$SESSION"
  ACTUAL_WORKERS=$((ACTUAL_WORKERS + 1))
done

echo "Created $ACTUAL_WORKERS windows"

# —— DISPATCH JOBS —— 
# Build commands for each window on-the-fly
idx=0
for cmd in "${CMDS[@]}"; do
  window=$(( idx % ACTUAL_WORKERS ))
  
  # Get current commands for this window
  eval "current_cmds=\${WINDOW_COMMANDS_${window}:-}"
  
  # Add new command
  if [[ -z "$current_cmds" ]]; then
    eval "WINDOW_COMMANDS_${window}=\"$cmd\""
  else
    eval "WINDOW_COMMANDS_${window}=\"$current_cmds ; $cmd\""
  fi
  
  idx=$((idx + 1))
done

# Show distribution
echo "Command distribution:"
for ((i=0; i<ACTUAL_WORKERS; i++)); do
  eval "window_cmds=\${WINDOW_COMMANDS_${i}:-}"
  cmd_count=$(echo "$window_cmds" | grep -o ';' | wc -l)
  cmd_count=$((cmd_count + 1))
  echo "  Window $i: $cmd_count commands"
done

# Send the chained commands to each window
for ((i=0; i<ACTUAL_WORKERS; i++)); do
  eval "window_cmds=\${WINDOW_COMMANDS_${i}:-}"
  if [[ -n "$window_cmds" ]]; then
    tmux send-keys -t "$SESSION":$i "$window_cmds" C-m
  fi
done

echo "Jobs dispatched! Attaching to session..."
sleep 1

# —— ATTACH & WATCH —— 
tmux attach-session -t "$SESSION" 