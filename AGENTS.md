# Project Conventions

## Environment Management
- This project uses pixi for dependency management
- All commands should be executed within the pixi environment

## Command Execution
- Prepend the launcher command `pixi run --no-progress` to anything you wanna run. E.g. for running `python`, run `pixi run --no-progress python`; if you need to `cd` somewhere do `cd` first, e.g. `cd src/datasets/<name> && pixi run --no-progress snakemake` for running `snakemake` in a specific directory.
  - Exception: If your termnial prompt indicates that the `cellwhisperer` environment is loaded, you can directly use `cellwhisperer` command as well as `python`` with all dependencies/libraries, as well as the `cellwhisperer` command itself.
- If you encounter this issue: `/lib64/libgcc_s.so.1: version 'GCC_7.0.0' not found` then `import pyarrow` in the *first* row of the python code/script.

# Coding Conventions

## General

- Try to make concise and smart contributions that leverage the existing code base, rather than writing a lot of new/additional code.
- Don't add dedicated error handling or checks for whether data is present or not (it's good/acceptable that scripts crash in those cases).

## Snakemake

- Within snakemake-related scripts and notebooks, assume that files are provided and contain the expected content. Hence, no need for dedicated if-else or try-catch checks for whether information is there or not.
- Use variables from snakemake directly (instead of aliasing them early in the script)
- Write concise and clean code. Avoid blowing up code into too many functions and files, as snakemake scripts are usually limited in their complexity
