# Cellwhisperer
CellWhisperer project

For more information on project management, follow
https://github.com/epigen/cellwhisperer/wiki


## Install

TODO: declare how to run with docker. also indicate how to install llava

1. Run `git clone git@github.com:epigen/cellwhisperer.git --recurse-submodules`
  If you already cloned, but did not add the `--recurse-submodules` run the following:
  `git submodule update --init --recursive`
2. Ensure correct environment (conda flexible channel prio and CUDA >= 12)

```bash
cat `~/.condarc`
...
channel_priority: flexible
```

```bash
(base) mschae83@s0-n11:~/cellwhisperer$ nvidia-smi
Sun Mar 31 17:36:31 2024
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 470.199.02   Driver Version: 470.199.02   CUDA Version: 12.2     |
...
```

2. Install the environment
  ```
  mamba env create -f envs/main.yaml  # this already includes `pip install -e .`
  conda activate cellwhisperer
  ```
3. Develop :)

### Optional: Installing scGPT

If you want to use scGPT instead of Geneformer (default), install `scgpt` and `flash-attn` (they need pip installation flags to be installed properly) via this script.
```
bash envs/install_scgpt_after_env_creation.sh
```

Then manually download the scGPT model, as indicated in the shell script.


### Optional: Installing CELLxGENE

See [developer_guidelines](./modules/cellxgene/dev_docs/developer_guidelines.md). In short:

- Make sure you have npm (install via conda or apt-get)
- For older versions of npm, run `export NODE_OPTIONS=--openssl-legacy-provider` ([workaround](https://stackoverflow.com/questions/69692842/error-message-error0308010cdigital-envelope-routinesunsupported))
- Build the client and put static files in place: `make build-for-server-dev`
- Install from local files: `make install-dev`
- Install prereqs for client: `make dev-env`

## Run

### Reproduce analyses and plots

### Process your own datasets for CellWhisperer+CELLxGENE Explorer

Refer to `src/cellxgene_preprocessing/README.md` for details.

### Preparation of training data for CellWhisperer training

This is a resource-intensive endeavor and not fully automated. We provide the full processed datasets for your convenience (https://medical-epigenomics.org/papers/schaefer2024/data/datasets/archs4_metasra_full_data.h5ad and https://medical-epigenomics.org/papers/schaefer2024/data/datasets/cellxgene_census_full_data.h5ad).


If you want to generate these datasets yourself, first you need to download the GEO/SRA/ARCHS4 and the CELLxGENE Census datasets:

```
cd src/datasets/archs4_metasra
snakemake  # Note that this pipeline source code is not thoroughly tested and was only executed in an interactive (non-pipeline) fashion
```
and
```
cd src/datasets/cellxgene_census
snakemake
```

Then, the whole process of generating annotations and preparing the datasets for training is captured in a dedicated pipeline (requires a large number of GPU hours):

```
cd src/pre_training_processing
snakemake
```

### Train CellWhisperer embedding model

We rely on pytorch lightning, which significantly reduces boilerplate for a multitude of aspects, including
- logging
- training
- CLI args
- model sharing

Before training, make sure to read the [LightningCLI documentation](https://lightning.ai/docs/pytorch/stable/cli/lightning_cli.html) (the three Basics are enough). It's short and really helpful! E.g. to start a run do this:

```bash
cd cellwhisperer  # go to the project directory to line up all paths correctly
cellwhisperer fit --print_config > run_config.yaml
# configure run_config.yaml
cellwhisperer fit --config run_config.yaml
```

#### Important parameters

`wandb`: Whether to log to wandb and if so which run_name to use
`trainer.logger.log_model`: Upload model to WandB?
`trainer.fast_dev_run`: Name is self-explanatory. Super useful for debugging
`ckpt_path`: a path (to load a model, e.g. for resuming)

#### TODO Sweeps

To run sweeps, refer to [this README](./src/experiments/sweeps/README.md). You can run sweeps with the `single_cell_sweeping` tool

### TODO Train CellWhisperer LLM

### TODO Model Analyses and plots


## Folder structure

- data: Computationally non-reproducible, expensive, or painful to reproduce
- metadata: Computationally non-reproducible, e.g., sample annotation sheets, clinical annotation
- results: Can be reproduced with your scripts and pipelines
- resources: External, references, datasets and tools that are project inherent and can be reproduced or downloaded with your scripts and pipelines
- src (and all other directories needed to run the source code)

### src/

Consists of the `cellwhisperer` package and a series of 

- `cellwhisperer`:
- `pre_training_processing`:
- `post_clip_processing`:  # TODO split
- `llava`:
- `ablation`:
- `figures`:

### Code style

We use `blacken` for automated code formatting.

## Deploy
- `cd` to `hosting/home`
    - To deploy, run `docker compose up -d`
    - To rebuild the website, run `docker compose -f website-builder-compose.yml up`  # TODO make this part of the docker file and remove here

## Processing of new (single cell) datasets
- Make sure your dataset adheres to the prerequisites described in the [wiki](https://github.com/epigen/cellwhisperer/wiki/Datasets)
- Place the dataset into a folder `cellxgene/resources/<dataset_name>/read_count_table.h5ad`
- Go to `cellwhisperer/src/post_clip_processing` and run `/msc/home/mschae83/cellwhisperer/results/<dataset_name>/cellwhisperer_clip_v1/cellxgene.h5ad` (use the correct model name)
