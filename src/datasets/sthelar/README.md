# STHELAR Dataset Testing

## Quick Test
For fast pipeline testing with a single slide (heart_s0):

```bash
# Test with single slide
snakemake --cores 4 --conda-frontend mamba -s src/datasets/sthelar/Snakefile test

# Dry run to see what would be executed
snakemake --cores 4 --conda-frontend mamba -s src/datasets/sthelar/Snakefile test -n
```

## Full Pipeline
To process all slides:

```bash
snakemake --cores 4 --conda-frontend mamba -s src/datasets/sthelar/Snakefile
```

The test mode processes only the `heart_s0` slide, which is smaller and faster for validation.