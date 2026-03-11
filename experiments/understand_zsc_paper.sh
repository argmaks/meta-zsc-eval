pixi run python run.py \
    --config-name=understand_zsc_paper \
    --multirun \
    runner.model=gemini/gemini-2.5-flash \
    runner.context_filepaths='[papers/obl.pdf]','[papers/op.pdf]'
