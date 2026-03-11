import os

SYSTEM_TASK_EXTRA_TEMPLATE = """
{system_prompt}

{task}

{extra_instructions}
"""

EXPLAIN_ZSC_TEMPLATE = """
{system_prompt}

Explain in detail the following zero-shot coordination method: {zsc_method}.

{extra_instructions}
"""

PROMPT_TEMPLATE = """
{prompt}
"""

SYSTEM_TASK_EXTRA_METHOD_FROM_LATEX_TEMPLATE = """
{system_prompt}

{task}

{extra_instructions}



{method_description_header}

{method_description}

{method_description_footer}
"""



def assemble_prompt(template_name, tex_files=[], **kwargs):

    repo_root = os.environ.get("PROJECT_DIR")
    papers_dir = os.path.join(repo_root, "context_files", "papers")
    if tex_files:
        assembled_tex = ""

        for tex_file in tex_files:
            tex_file_path = os.path.join(papers_dir, tex_file)
            with open(tex_file_path, "r") as f:
                assembled_tex += f.read()
                assembled_tex += "\n\n"

        fenced_tex = f"```latex\n{assembled_tex}\n```"
    
    # Use a dictionary to map template names to their corresponding templates
    templates = {
        "SYSTEM_TASK_EXTRA_TEMPLATE": SYSTEM_TASK_EXTRA_TEMPLATE,
        "EXPLAIN_ZSC_TEMPLATE": EXPLAIN_ZSC_TEMPLATE,
        "PROMPT_TEMPLATE": PROMPT_TEMPLATE,
        "SYSTEM_TASK_EXTRA_METHOD_FROM_LATEX_TEMPLATE": SYSTEM_TASK_EXTRA_METHOD_FROM_LATEX_TEMPLATE,
    }
    try:
        template = templates[template_name]
    except KeyError:
        return None

    # Convert None values to empty strings to avoid "None" appearing in the output
    formatted_kwargs = {k: "" if v is None else v for k, v in kwargs.items()}
    
    if template_name == "SYSTEM_TASK_EXTRA_METHOD_FROM_LATEX_TEMPLATE":
        return template.format(
            **formatted_kwargs,
            method_description=fenced_tex,
        )
    else:
        return template.format(**formatted_kwargs)

if __name__ == "__main__":
    print(assemble_prompt("EXPLAIN_ZSC_TEMPLATE", system_prompt=None, task=None, extra_instructions=None, zsc_method="Other-play"))
    # print(assemble_prompt("SYSTEM_TASK_EXTRA_METHOD_FROM_LATEX_TEMPLATE", tex_files=["op/abstract.tex", "op/zsc.tex", "op/op.tex"], system_prompt="system prompt", task="task", extra_instructions="extra instructions", method_description_header=None, method_description_footer=None))