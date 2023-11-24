This directory contains the snakemake-pipeline to curate our annotations (for now via the OpenAI API)

# API key

Provide your OpenAI API key via the environment variable `OPENAI_API_KEY`. E.g.:

```bash
export OPENAI_API_KEY=sk-ctiercntie
```

# Nginx

`conda activate nginx`

Start with `nginx -c $HOME/single-cellm/src/annotation_processing/pipeline/nginx/load_balancer.conf`

After adding server, run `nginx -s reload -p  /msc/home/mschae83/miniconda3/envs/nginx/` to reload the config on the fly :)
