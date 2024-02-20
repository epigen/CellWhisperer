This directory contains the snakemake-pipeline to curate our annotations (for now via the OpenAI API)

# API key

Provide your OpenAI API key via the environment variable `OPENAI_API_KEY`. E.g.:

```bash
export OPENAI_API_KEY=sk-ctiercntie
```

# Snakemake

Run like below. Seems hacky but works. It batches the relevant rule, to prevent snakemake-stalling (due to large DAG size)

```bash
for i in $(seq 100); do
  echo "Running batch $i/100"
  # NOTE: I needed to add `-R process_annotation_local` as some sort of a workaround
  SNAKEMAKE_PROFILE= snakemake -j12 -R process_annotation_local --rerun-triggers mtime --batch aggregate_processed=$i/100
done

```

# LLM server

## oobabooga

With this configuration (8 bit quantization), needs an 80GB GPU

```bash
cd /msc/home/mschae83/text-generation-webui
./start_linux.sh  # see `git diff`, as I configured everything

# alternative:
# conda activate textgen
# python server.py --listen-host 0.0.0.0 --listen --api --api-port 5000 --model mixtral-8x7b-instruct-v0.1.Q8_0.gguf --public-api --api-key OdNFpEua0T5TNdd0

```

## nginx
`conda activate nginx`

Start with `nginx -c $HOME/cellwhisperer/src/annotation_processing/pipeline/nginx/load_balancer.conf`

After adding server, run `nginx -s reload -p  /msc/home/mschae83/miniconda3/envs/nginx/` to reload the config on the fly :)
