# meta-zsc

Evaluating how robust Zero Shot Coordination methods are to implementation details by sampling implementations via a custom LLM Coding agent.

The agent is given a paper in .tex or .pdf and asked to implement the method described, train it and iterate until satisfactory. We then subsample the correct implementations and evaluate them in cross-play across implementations.

## Installation

Install pixi
```
curl -fsSL https://pixi.sh/install.sh | sh
```

You can install the package in development mode using:

```bash
pixi run postinstall # installs meta-zsc 
pixi run -e analysis postinstall # installs meta-zsc 
pixi run -e analysis install_jaxmarl # installs meta-zsc 
pixi run -e jax postinstall # installs meta-zsc  
pixi run -e jax install_jaxmarl # installs jaxmarl
```

If running on a Linux, CUDA enabled node instead of the `jax` environment use `jax-cuda` to be able to train on the GPUs.
```bash
pixi run -e jax-cuda postinstall # installs meta-zsc  
pixi run -e jax-cuda install_jaxmarl # installs jaxmarl
```

Install direnv
```
curl -sfL https://direnv.net/install.sh | bash
```
And follow instructions (add the eval script to your bashrc / zshrc)

Install tmux
```
# TODO
```

Create `/.envrc` local using template `.envrc.EXAMPLE.local`

## Repository structure

```
meta-zsc/
├── configs/                    # Experiment configurations (YAML files)
│   ├── default.yaml           # Default configuration template
│   ├── general_zsc_knowledge.yaml   # LLM knowledge assessment
│   ├── aider_scaffold*.yaml   # Automated coding with Aider
│   └── ...                    # Other experiment configs
├── context_files/             # Core ZSC environments and algorithms
│   ├── envs/                  # Multi-agent game environments
│   │   ├── cat_dog/           # Cat-Dog coordination game
│   │   └── lever_game/        # Two-player lever selection game
│   ├── algos/                 # Algorithm implementations by environment
│   │   ├── cat_dog/           # Q-learning variants for Cat-Dog game
│   │   └── lever_game/        # Q-learning variants for Lever game
│   ├── evaluate/              # Evaluation scripts for trained agents
│   └── papers/                # Reference papers (ZSC methods)
├── runners/                   # Experiment execution backends
│   ├── query_llm.py          # Direct LLM querying with prompt templates
│   ├── aider_scaffold.py     # Automated coding using Aider AI assistant
│   ├── initiate_pair_coding.py # Setup for interactive coding sessions
│   └── train_algo.py         # Algorithm training workflows
├── prompts/                   # Prompt template system
│   └── task_prompts.py       # Template definitions and assembly
├── experiments/               # Bash scripts for running experiments
├── analysis/                  # Analysis and evaluation utilities
├── samples/                   # Generated experiment outputs and workspaces
├── meta_zsc/                  # Main Python package
└── run.py                     # Main entry point (Hydra-based)
```

## How to run

A single run:

``` 
pixi run python run.py --config-name=<name of the .yaml file configuring the experiment>
```

A single run with overrides (example)

```
pixi run python run.py --config-name=<name of the .yaml file configuring the experiment> runner.model=gemini-2.5-pro
```

A hydra multi run over override parameters. Note: this will run sequentially in a single terminal window.

```
pixi run python run.py -m --config-name=<name of the .yaml file configuring the experiment> runner.model=gemini-2.5-pro,openai/o3
```

A full experiment run. Note: this will run in parallel across many tmux terminal windows. The number of parallel workers can be specficied in the experiment shell script.

```
experiments/<experiment_name>.sh
```

You will automatically get attached to the tmux session. `Ctrl + b d` to detach from the session. Also see the Tmux Cheat Sheet & Quick Reference https://tmuxcheatsheet.com.

Run 
```
tmux kill-server
```
after running an experiment to close the session.

## Samples

Sampled implementations are stored in /samples. Each sample typically contains a config file used to create the sample, a results file with summarized metrics for the run, the implemented python scripts along with context files and other traces of the implementation process.

## Analysis

Scripts in `/analysis` produce the XXP scores as well as the auxillary statistics. Run the analysis scripts using the "analysis" environment.
