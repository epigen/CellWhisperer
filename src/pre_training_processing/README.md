This directory contains the snakemake-pipeline to curate our annotations (for now via the OpenAI API)

# API key

Provide your OpenAI API key via the environment variable `OPENAI_API_KEY`. E.g.:

```bash
export OPENAI_API_KEY=sk-ctiercntie
```

# Snakemake

Just run:

```bash
SNAKEMAKE_PROFILE=muwhpc_slurm snakemake
```

This automatically splits the heavy load of annotation processing into 256 manageable splits, which are then processed 1 by 1 (and/or in parallel)

# LLM server

## oobabooga

With this configuration (5 bit quantization), needs a 40GB GPU

```bash
cd /msc/home/mschae83/text-generation-webui
./start_linux.sh  # see `git diff`, as I configured everything

# alternative:
# conda activate textgen
# Note https://www.reddit.com/r/LocalLLaMA/comments/18jxehq/guide_to_run_mixtral_correctly_i_see_a_lot_of/
# python server.py --listen-host 0.0.0.0 --listen --api --api-port 5000 --model mixtral-8x7b-instruct-v0.1.Q5_K_M.gguf --public-api --api-key OdNFpEua0T5TNdd0
```

## nginx
`conda activate nginx`

Start with `nginx -c $HOME/cellwhisperer/src/pre_training_processing/llm/nginx_load_balancer.conf`

After adding server, run `nginx -s reload -p  /msc/home/mschae83/miniconda3/envs/nginx/` to reload the config on the fly :)
