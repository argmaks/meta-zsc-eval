import os

class AiderScaffoldRunner:

    def __init__(self, 
                 model,
                 aider_read_context_fnames = [],
                 aider_edit_context_fnames = [],
                 llm_instructor = False,
                 llm_instructor_context_fnames = [],
                 feedback_loop_iters = 2,
                 feedback_prompt = None,
                 aider_run_script = None,
                 ):
        self.model = model
        self.aider_read_context_fnames = aider_read_context_fnames
        self.aider_edit_context_fnames = aider_edit_context_fnames
        self.aider_run_script = aider_run_script
        self.llm_instructor = llm_instructor
        self.llm_instructor_context_fnames = llm_instructor_context_fnames
        self.feedback_loop_iters = feedback_loop_iters
        self.feedback_prompt = feedback_prompt

    def run(self, prompt_content, repo_root, workspace_dir):


        if self.llm_instructor == True:
            from runners.query_llm import QueryLLMRunner
            llm_instructor_runner = QueryLLMRunner(model=self.model, context_filepaths=self.llm_instructor_context_fnames)
            response = llm_instructor_runner.run(prompt_content, repo_root, workspace_dir)
            instruction = response["runner_response"]

        else:
            instruction = prompt_content


        
        inchat_aider_commands = []

        # add read-only command
        read_command = ' '.join(["/read-only"] + self.aider_read_context_fnames)
        inchat_aider_commands.append(read_command)

        # add edit command
        edit_command = ' '.join(["/add"] + self.aider_edit_context_fnames)
        inchat_aider_commands.append(edit_command)

        # add instruction command
        instruction_command = f"/code {instruction.replace('\n', '\\n')}"
        inchat_aider_commands.append(instruction_command)

        # add run script command
        run_script_command = f"/run pixi run -e jax python {self.aider_run_script}"
        inchat_aider_commands.append(run_script_command)

        # feedback loop commands
        feedback_loop_commands = []
        for i in range(self.feedback_loop_iters):
            feedback_prompt_command = f"/code {self.feedback_prompt}"
            feedback_loop_commands.append(feedback_prompt_command)
            feedback_loop_commands.append(run_script_command)

        inchat_aider_commands.extend(feedback_loop_commands)

        inchat_aider_commands.append("/exit")

        # assemble inchat_aider_commands from strings in inchat_aider_commands and write to a file inchat_aider_commands.txt where each string is on a new line
        with open(os.path.join(workspace_dir, "inchat_aider_commands"), "w") as f:
            for command in inchat_aider_commands:
                f.write(command + "\n")

        # assemble the run aider command
        run_aider_command = f"aider --model {self.model} --map-tokens 0 --no-git --chat-history-file .aider.chat.history.md --input-history-file .aider.input.history --yes-always --load inchat_aider_commands"

        # run the run_aider_command from the workspace_dir
        os.system(f"cd {workspace_dir} && {run_aider_command}")

        return {"runner_response": "Aider scaffold run successfully."}
