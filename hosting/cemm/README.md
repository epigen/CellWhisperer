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

## Docker Hub

The cellwhisperer/spotwhisperer image is backed up to docker hub, and can respectively be restored from there. In order to do so you need to be logged in to docker hub, *not* the red hat registry which podman auto-logs you into.

`docker login docker.io`
username: jburto26
token: dckr_pat_wNFKfvpc9B_d2235P-vCxnj3P1s

This is just a public access token. If it doesn't work see #782 for how to regenerate.

## `/var` Size

If `/var/` becomes too full, e.g., because of image building, everything will fail. How does this happen, well, because of the cellwhisp setup, /home is too small for building and so `/home/jburton/.local/share` is symlinked to `/var/tmp/jburton/symlinked/containers`. Thus container storage ends up on var. The easist way to free space is to `mv` the containers folder to something else and then set up the sylinks again. Everything will be rebuilt. You may have to `chown` and `chmod` the previous containers directory to get it into shape where you can delete it. Time often helps here.

## Maintenance

If the website needs to go offline for a larger amount of time due to maintenance, rename the file `static/maintenance.hidden.html` to `static/maintenance.html` and keep the nginx server running. Once maintenance is done, rename the file back to its original name again.

## Troubleshooting: Complete Rebuild

If you need to completely rebuild the container environment:

1. Stop the container runtime (podman/docker)
2. Remove old container data and volumes
3. Rebuild images: `docker compose build`
4. Start services: `docker compose up -d`

**Note:** Specific cleanup steps will depend on your container runtime configuration.

## Docker Nginx Sidecar

To support dynamic scaling/loading of user supplied datasets we use [docker-gen](https://hub.docker.com/r/jwilder/docker-gen) to dynamically update the nginx configuration when new containers are started. This requires an additional container `docker-gen` which runs on the same network and monitors for changes to the running containers. On detecting a change it will re-write the nginx configuration template in `nginx.tmpl` to insert container metadata. For existing containers this metadata comes from the `docker-compose` file. Specifically the `PROXY_LOCATION_PATH: "lc5"` and `GENERATE_PROXY_LOCATION: "true"` keys. This cause a new entry to be inserted into the nginx config to proxy requests to the PROXY_LOCATION_PATH to this container. This metadata in the future will also come from the podman python library.

Thus, `nginx.tmpl` now contains a template for an nginx sub-configuration that is re-written and loaded into `/etc/nginx/conf.d/` and `nginx.conf` contains only logging boilerplate and the directive to include these dynamic files.



