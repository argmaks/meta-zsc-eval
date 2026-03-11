from pathlib import Path

class InitiatePairCodingRunner:

    def __init__(self, model: str, coding_tool: str, env: str, algo: str, algo_config: dict, tests: dict):
        self.model = model
        self.coding_tool = coding_tool
        self.env = env
        self.algo = algo
        self.algo_config = algo_config
        self.tests = tests

    def run(self, prompt_content: str, repo_root: Path=None, workspace_dir: Path = None):
        
        print(f"Running pair coding with {self.model} and {self.coding_tool}")
        print(f"Prompt: {prompt_content}")
        
        print("Command to initiate pair coding:")

        if self.coding_tool == "aider":
            print(f"cd {workspace_dir} && aider --model {self.model} --map-tokens 0 --no-git --chat-history-file .aider.chat.history.md --input-history-file .aider.input.history --yes-always --analytics-disable --no-check-update")
        elif self.coding_tool == "cursor":
            print(f"")
        else:
            raise NotImplementedError(f"Coding tool {self.coding_tool} not implemented")

        return {"runner_response": "Initiated pair coding."}

