from pathlib import Path
import hydra
from omegaconf import OmegaConf
import shutil
import json
from datetime import datetime
from prompts.task_prompts import assemble_prompt
import os

@hydra.main(version_base=None, config_path="configs", config_name="default")
def main(cfg):
    
    # print the configuration
    print(OmegaConf.to_yaml(cfg))

    repo_root = Path(os.environ["PROJECT_DIR"])

    if cfg.workspace_dir is None:

        # create a new directory for the sample in samples
        workspace_dir = repo_root / "samples" / cfg.experiment.name

        # append timestamp to the sample directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        workspace_dir = workspace_dir / timestamp

        workspace_dir.mkdir(parents=True, exist_ok=True)
        print(f"Directory created at: {workspace_dir}")

    else:
        workspace_dir = Path(cfg.workspace_dir)

    # write the configuration into a file in the sample directory
    cfg_path = workspace_dir / "config.yaml"
    with open(cfg_path, "w") as f:
        OmegaConf.save(cfg, f)

    # get the list of repo template files from the config
    if cfg.env is not None: 
        env_dir = repo_root / "context_files" / "envs" / cfg.env # (e.g "lever_game")
    if cfg.algo is not None:
        algo_dir = repo_root / "context_files" / "algos" / cfg.env / cfg.algo # (e.g "q_learning_1")
    extra_workspace_files = cfg.extra_workspace_files # (e.g ["papers/op.pdf"])

    # copy over the contents of the env and algo directories into the workspace directory keeping the subdirectory structure 
    if cfg.env is not None:
        for item in os.listdir(env_dir):
            if item.startswith('.') or item.startswith('_'):
                continue
            src_path = env_dir / item
            dst_path = workspace_dir / item
            if src_path.is_dir():
                shutil.copytree(src_path, dst_path)
            else:
                shutil.copy(src_path, dst_path)
    if cfg.algo is not None:
        for item in os.listdir(algo_dir):
            if item.startswith('.') or item.startswith('_'):
                continue
            src_path = algo_dir / item
            dst_path = workspace_dir / item
            if src_path.is_dir():
                shutil.copytree(src_path, dst_path)
            else:
                shutil.copy(src_path, dst_path)


    # copy over the extra workspace files without the subdirectory structure 
    for file in extra_workspace_files:
        shutil.copy(repo_root / "context_files" / file, workspace_dir / file.split("/")[-1])

   
    # assemble the prompt
    prompt_content = assemble_prompt(
        cfg.prompt.template_name,
        **cfg.prompt.contents
    )


    # instantiate the runner
    runner = hydra.utils.instantiate(cfg.runner)

    # run the runner
    result = runner.run(prompt_content, repo_root, workspace_dir)
        
    # save the result to json, appending to dictionary if file exists
    result_path = workspace_dir / "results.json"
    
    # Load existing results if file exists
    if result_path.exists():
        try:
            with open(result_path, 'r') as f:
                all_results = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            # If file is corrupted or empty, start fresh
            all_results = {}
    else:
        all_results = {}
    
    # Add current result to the dictionary
    all_results = all_results | result
    
    # Save updated results back to file
    with open(result_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"Experiment completed. Results saved to: {result_path}")
    return result

if __name__ == "__main__":
    main()