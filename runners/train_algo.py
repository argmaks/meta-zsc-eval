import os
from pathlib import Path

class TrainAlgoRunner:
    def __init__(self, script_name, **kwargs):
        self.script_name = script_name
        self.kwargs = kwargs

    def run(self, prompt_content: str, repo_root: Path=None, workspace_dir: Path = None):
        
        if self.script_name is not None:
            os.system(f"pixi run python -e jax-cuda {workspace_dir}/{self.script_name}")

        return {"runner_response": "Run successfully."}
        
        