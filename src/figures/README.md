Figures pipeline
======================

This pipeline generates all plots (where possible) for our manuscript (Note: some of the panels are screenshots that were directly captured from the web-app.)

Simply run `snakemake`. Note that this may require a substantial amount of RAM and CPU. We recommend running it on a cluster (e.g. using the snakemake SLURM integration)

You might be requested to login to huggingface to be able to download the Mistral-7B model. Simply follow the instructions printed in the command line. The `huggingface-cli` tool is installed in the `cellwhisperer` environment.
