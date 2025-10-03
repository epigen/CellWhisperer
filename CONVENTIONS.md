# Project Conventions

## Environment Management
- This project uses pixi for dependency management
- All commands should be executed within the pixi environment

## Command Execution
- Always use `pixi run --no-progress` prefix when running Python scripts or other commands
- Examples:
  - `pixi run python script.py` instead of `python script.py`
  - `cd src/datasets/<name> && pixi run snakemake`
  - `pixi run pytest`


# Coding Conventions

## Snakemake

- Within snakemake-related scripts and notebooks, assume that files are provided and contain the expected content. Hence, no need for dedicated if-else or try-catch checks.
- Use variables from snakemake directly (instead of aliasing them early in the script)
- Write concise and clean code. Avoid blowing up code into too many functions and files, as snakemake scripts are usually limited in their complexity
