# Production Hosting Configuration

This folder contains an example configuration for deploying CellWhisperer in a production environment. This setup is based on the deployment used at CeMM but can be adapted for other institutions.

## Deployment

Upon changes/deployment, it is recommended that you cycle the containers by running `./cellwhisperer_cycle.sh`. This ensures that the network is set up correctly and the container environment is kept clean. Additionally, rebuild the image using `docker compose build`.

Alternatively, consult that same file to determine which commands to run.

## Storage Setup

The setup requires that datasets and models are accessible via mounted directories. For each dataset service, you need a corresponding dataset file. For example, for a dataset named `hcaorganoids_normal_organoid`, the following file should exist:

```
/path/to/results/hcaorganoids_normal__organoid/cellwhisperer_clip_v1/cellxgene.h5ad
```

**Note:** Update the volume mounts in `docker-compose.yml` to point to your actual data directories.

## Model and Resource Setup

The resources folder and model directories (e.g., `.../results/<model_name>/cellwhisperer_clip_v1/...`) should be populated with the required models.

Models can be fetched using `snakemake`:

1. Ensure `snakemake` is installed: `mamba install -c bioconda snakemake=7.15.2`
2. Fetch required models: `snakemake -j1 models` (run from the `src` folder in the repository root)

## Maintenance

If the website needs to go offline for a larger amount of time due to maintenance, rename the file `static/maintenance.hidden.html` to `static/maintenance.html` and keep the nginx server running. Once maintenance is done, rename the file back to its original name again.

## Troubleshooting: Complete Rebuild

If you need to completely rebuild the container environment:

1. Stop the container runtime (podman/docker)
2. Remove old container data and volumes
3. Rebuild images: `docker compose build`
4. Start services: `docker compose up -d`

**Note:** Specific cleanup steps will depend on your container runtime configuration.

