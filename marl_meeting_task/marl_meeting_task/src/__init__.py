import os

# Redirect `marl_meeting_task.src` package to the repository's top-level `src/` folder
__path__.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

