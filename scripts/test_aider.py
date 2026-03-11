# note: run with the aider environment

from aider.coders import Coder
from aider.models import Model

# This is a list of files to add to the chat
fnames = []

model = Model("gpt-4-turbo")

# Create a coder object
coder = Coder.create(main_model=model, fnames=fnames)

# This will execute one instruction on those files and then return
coder.run("make a script that prints hello world")