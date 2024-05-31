Figures pipeline
======================

This pipeline generates all plots (where possible) for our manuscript (Note: some of the panels are screenshots that were directly captured from the web-app.)

Simply run `snakemake`. Note that this may require a substantial amount of RAM and CPU. We recommend running it on a cluster (e.g. using the snakemake SLURM integration)

In order to run the GPT-4-judge evaluation you need to the set the `OPENAI_API_KEY` environment variable (estimated cost: ~3 USD).
