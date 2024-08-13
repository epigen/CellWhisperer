# Hosting @ CeMM

This folder contains the configuration necessary for deploying CellWhisperer on CeMM
infrastructure. It is recommended that you cycle the cellwhisperer containers by
running `./cellwhisperer_cycle.sh`. This will ensure that the network is set up correctly
and that the container enivronment is kept clean.

Alternatively you can consult the same file to determine which commands to run.

Note that the setup requires that `/nobackup/lab_bock/` is mounted (e.g., via sshfs) so
that `/nobackup/lab_bock/projects/cellwhisperer/results` and `resources` is accessible.

The resources folder should be populated with the required models.

For each model service therer should be a corresponding model file. For example for 
hcaorganoids_normal_organoid there should be a model:

```
/nobackup/lab_bock/projects/cellwhisperer/results/hcaorganoids_normal__organoid/cellwhisperer_clip_v1/cellxgene.h5ad
```

In general for each model service you want to run a model file should be at:

```
.../results/<model_name>/cellwhisperer_clip_v1/...`
```

If these are not present they can be fetched using `snakemake`. First you will need to
 install `snakemake` if you haven't already:

```
mamba install -c bioconda snakemake=7.15.2
```

You can then fetch the required modules by running `snakemake models` from the `src` folder
in the repository root.
