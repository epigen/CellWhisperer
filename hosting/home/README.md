# Home server


Here I collect all the files required for my home server

```bash
docker compose up -d
# Attach or exec
docker compose exec conda /bin/bash
```
Note: nvidia-docker is deprecated

## Caveats

- nginx may point to wrong containers, as their IPs change when they are restarted. Make sure to restart nginx upon changes
