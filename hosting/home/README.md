# Home server


Here I collect all the files required for my home server

```bash
docker compose up -d
# Attach or exec
docker compose exec conda /bin/bash
```
Note: nvidia-docker is deprecated

## Rebuild web client

To rebuild the website, run `docker compose -f website-builder-compose.yml up`
