# Hosting @ CeMM

This folder contains the configuration necessary for deploying CellWhisperer on CeMM
infrastructure. Upon changes/deployment, it is recommended that you cycle the cellwhisperer containers by
running `./cellwhisperer_cycle.sh`. This will ensure that the network is set up correctly
and that the container environment is kept clean.

Alternatively you can consult that same file to determine which commands to run.

Note that the setup requires that `/nobackup/lab_bock/` is mounted (e.g., via sshfs) so
that `/nobackup/lab_bock/projects/cellwhisperer/results` and `resources` are accessible.

For each dataset service there should be a corresponding dataset file. For example for
hcaorganoids_normal_organoid the following file should exist:

```
/nobackup/lab_bock/projects/cellwhisperer/results/hcaorganoids_normal__organoid/cellwhisperer_clip_v1/cellxgene.h5ad
```

The resources folder as well as the cellwhisperer model folders (e.g. `.../results/<model_name>/cellwhisperer_clip_v1/...`)
should be populated with the required models.

These can be fetched using `snakemake`:
First you would need to ensure that `snakemake` is installed (`mamba install -c bioconda snakemake=7.15.2`)

You can then fetch the required modules by running `snakemake models` from within the `src` folder
in the repository root.

## Maintenance

If the website needs to go offline for a larger amount of time due to maintenance, rename the file `static/maintenance.hidden.html` to `static/maintenance.html` and keep the nginx server running. Once maintenance is done, rename the file back to its original name again.
