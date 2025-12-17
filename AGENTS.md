# Project Conventions

## Environment Management
- This project uses pixi for dependency management
- All commands should be executed within the pixi environment

## Command Execution
- You should be in the `cellwhisperer` environment, being able to use python with all dependencies/libraries, as well as the `cellwhisperer` command itself.
- If you encounter this issue: `/lib64/libgcc_s.so.1: version 'GCC_7.0.0' not found` then `import pyarrow` in the *first* row of the python code/script.

# Coding Conventions

## General

- Try to make concise and smart contributions that leverage the existing code base, rather than writing a lot of new/additional code.

## Snakemake

- Within snakemake-related scripts and notebooks, assume that files are provided and contain the expected content. Hence, no need for dedicated if-else or try-catch checks.
- Use variables from snakemake directly (instead of aliasing them early in the script)
- Write concise and clean code. Avoid blowing up code into too many functions and files, as snakemake scripts are usually limited in their complexity
