# Project Conventions

## Environment Management
- This project uses uv for dependency management
- All commands should be executed within the uv environment

## Command Execution
- Always use `uv run` prefix when running Python scripts or other commands
- Examples:
  - `uv run python script.py` instead of `python script.py`
  - `cd src/datasets/<name> && uv run snakemake`
  - `uv run pytest`

