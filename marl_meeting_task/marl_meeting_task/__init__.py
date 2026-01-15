import os

# Allow imports like `marl_meeting_task.src...` by pointing the package search
# path to the repository's `src/` directory.
__path__.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

