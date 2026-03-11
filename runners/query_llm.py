from litellm import completion
import time
import base64
from pathlib import Path
from typing import List
import os

class QueryLLMRunner:

    def __init__(self, model="gemini/gemini-2.5-flash", reasoning_effort=None, temperature=None, max_tokens=None, response_format=None, tools=None, tool_choice=None, context_filepaths=[]):
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.response_format = response_format
        self.tools = tools
        self.tool_choice = tool_choice
        self.context_filepaths = context_filepaths


    def run(self, prompt_content: str, repo_root: Path=None, workspace_dir: Path = None):
        
        # prepare the query
        content = prompt_content
        model = self.model
        reasoning_effort = self.reasoning_effort

        project_root = Path(os.getenv("PROJECT_DIR"))
        context_filepaths = [project_root / "context_files" / context_filepath for context_filepath in self.context_filepaths]
        
        start_time = time.time()

        content = [{"type": "text", "text": prompt_content}]
        # assemble the messages based on whether context files are provided
        if len(context_filepaths) == 1:
            files = self._read_context_files(context_filepaths)
            content = content + files
            messages = [{"role": "user", "content": content}]
        elif len(context_filepaths) > 1:
            raise ValueError("Only one context file is supported for now.")
        else:
            messages = [{"role": "user", "content": content}]
        
        # invoke the model
        response = completion(
            model=model,
            reasoning_effort=reasoning_effort,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            response_format=self.response_format,
            tools=self.tools,
            tool_choice=self.tool_choice,
        )
        
        end_time = time.time()
        # import pdb; pdb.set_trace()
        # process the response
        response_text = response['choices'][0]['message']['content']
        response_metadata = {
            "model": model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "execution_time": end_time - start_time,
            "usage": dict(response.usage) if hasattr(response, 'usage') else None
        }
        # import pdb; pdb.set_trace()
        # return the response as a dictionary
        result = {
            "runner_response": response_text,
        }
        return result

    def _read_context_files(self, context_filepaths: List[Path]):

        files = []
        for i, context_filepath in enumerate(context_filepaths):
            with open(context_filepath, "rb") as f:
                file_content = f.read()
                base64_content = base64.b64encode(file_content).decode('utf-8')
                file_data = f"data:application/pdf;base64,{base64_content}"
                file = {
                    "type": "file",
                    "file": {
                        "file_data": file_data,
                        # "filename": context_filepath.name, 
                    },
                }
                files.append(file)
        return files

if __name__ == "__main__":


    context_filepaths = ["papers/op.pdf"]
    runner = QueryLLMRunner(model="gemini/gemini-2.5-flash")
    result_zero = runner.run("What is Zero-Shot Coordination in one sentence?")
    runner = QueryLLMRunner(model="gemini/gemini-2.5-flash", context_filepaths=context_filepaths)
    result_one = runner.run("Summarize the attached paper in one sentence.")
    print(result_zero)
    print(result_one)