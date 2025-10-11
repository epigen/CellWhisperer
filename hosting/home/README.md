# Home Server Deployment

This directory contains configuration files for deploying CellWhisperer on a personal server or workstation.

## Quick Start

```bash
docker compose up -d
# Attach or exec to interact with containers
docker compose exec conda /bin/bash
```

**Note:** nvidia-docker is deprecated. Use `docker --gpus` flag instead (as shown in docker-compose.yml).

## Rebuild web client

To rebuild the website, run `docker compose -f website-builder-compose.yml up`

## Caveats

- nginx may point to wrong containers, as their IPs change when they are restarted. Make sure to restart nginx upon changes
