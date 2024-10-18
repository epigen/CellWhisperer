# CellWhisperer

CellWhisperer is a multimodal AI model combining transcriptomics with natural language to enable intuitive interaction with scRNA-seq datasets. This repository contains source code for dataset generation, training, inference and the CellWhisperer web tool, which is based on the CELLxGENE Explorer. The [project website](http://cellwhisperer.bocklab.org/) hosts the web tool with several example datasets as well as a short video tutorial.

An early version of CellWhisperer has been accepted as [Spotlight paper at the ICLR 2024 MLGenX workshop](https://openreview.net/forum?id=yWiZaE4k3K). The full paper can be found on [bioRxiv](https://www.biorxiv.org/content/10.1101/2024.10.15.618501v1).

##### Table of Contents

- [Install](#install)
- [Run paper analyses](#run)
- [Folder structure](#structure)
- [Analyze your own datasets](#analyze)


<a name="install"/>
  
## Install

To obtain this repository, run `git clone https://github.com/epigen/cellwhisperer.git --recurse-submodules`.

*You do need the submodules*, so if you already cloned, but without the `--recurse-submodules` flag, run the following: `git submodule update --init --recursive`

CellWhisperer can be run through conda and docker. Both options take about 15 minutes to set up. The versioned packages installed in the environments are defined in the files in the `envs` folder.

### Install via conda/pip

1. Ensure correct environment (conda flexible channel prio and CUDA >= 12)

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

2. Install the environments
  ```
  mamba env create -f envs/main.yaml  # name: cellwhisperer    this already includes `pip install -e .`
  mamba env create -f envs/llava.yaml  # name: llava    this already includes `pip install -e ../modules/LLaVA/[train]`
  mamba env create -f envs/llama_cpp.yaml  # name: llama_cpp

  conda activate cellwhisperer
  ```
3. Run the app or analyze your datasets :)

#### Optional: Installing scGPT

If you want to use scGPT instead of Geneformer (default), install `scgpt` and `flash-attn` (they need pip installation flags to be installed properly) via this script.

```
bash envs/install_scgpt_after_env_creation.sh
```

Then manually download the scGPT model, as indicated in the shell script.

Note: You might need to install gcc and gxx (e.g. v9.5) if you don't have them. Note that if you install them with conda might lead to [issues](https://github.com/conda/conda/issues/6945) with snakemake.

#### Optional: Installing CELLxGENE single cell browser web app

1. `conda activate cellwhisperer` (provides `npm`, which is required for (2))
2. `cd modules/cellxgene && make build-for-server-dev` (builds the client)

For details, refer to the [developer_guidelines](./modules/cellxgene/dev_docs/developer_guidelines.md).

<a name="run"/>

### Install within Docker

You can also install and use CellWhisperer within docker, which includes all installation steps above (including cellxgene installation):

```bash
docker build -t cellwhisperer .
docker run --gpus all -it --volume .:/opt/cellwhisperer cellwhisperer bash  # also works without GPUs
conda activate cellwhisperer
```

Note that this container mounts the project directory as volume (`--volume .:/opt/cellwhisperer`) in the container (such that code modifications are visible in the container). Consider mounting also a `resources` and `results` directory to `/opt/cellwhisperer/resources` and `/opt/cellwhisperer/results`, as these are source and target directories when processing datasets (see section [Analyze your own datasets](#analyze) below).


## Run

### Reproduce manuscript analyses and plots

We provide all our validations and analyses in a single pipeline, (re)producing all (*) plots in our paper.

Note that due to the high computational cost, this pipeline relies on some precomputed files, which are downloaded from our server as part of the pipeline. Nevertheless computing all the analyses will require a considerable amount of storage (~1TB), RAM (up to 1TB), GPU and time (2 days) resources.

To run the pipeline, execute

```bash
cd src
snakemake --use-conda
```

(*) Some interactive analyses/screenshots were performed directly in the CELLxGENE CellWhisperer browser integration and are not reproduced by the pipeline. Also note that some analyses (e.g. Extended Data Figure 3) depend on GPT-4 and the availability of an OpenAI API key.

Refer to `src/figures/README.md` for further details.


### Preparation of training data for CellWhisperer training

This is a resource-intensive endeavor and not fully automated. We provide the full processed datasets for your convenience (https://medical-epigenomics.org/papers/schaefer2024/data/datasets/archs4_geo/full_data.h5ad and https://medical-epigenomics.org/papers/schaefer2024/data/datasets/cellxgene_census/full_data.h5ad).


If you want to generate these datasets yourself, first you need to download the GEO/SRA/ARCHS4 and the CELLxGENE Census datasets:

```
cd src/datasets/archs4_geo
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

The config used to train is located at `src/cellwhisperer_clip_v1.yaml`

#### Important parameters

`wandb`: Whether to log to wandb and if so which run_name to use
`trainer.logger.log_model`: Upload model to WandB?
`trainer.fast_dev_run`: Name is self-explanatory. Super useful for debugging
`ckpt_path`: a path (to load a model, e.g. for resuming)

### Train CellWhisperer LLM

1. Go to `src/llava`
2. Run `snakemake`

Note 1: The pipelines includes code to generate the datasets. Since this takes a considerable amount of time and computational resources, we recommend downloading our provided data set. (automatically done by the main pipeline in `src`).
Note 2: You might be requested to login to huggingface to be able to download the Mistral-7B model. Simply follow the instructions printed in the command line. The `huggingface-cli` tool is installed in the `cellwhisperer` environment. 

<a name="structure"/>

## Folder structure

- data: Computationally non-reproducible, expensive, or painful to reproduce
- results: Can be reproduced with your scripts and pipelines
- resources: External, references, datasets and tools that are project inherent and can be reproduced or downloaded with your scripts and pipelines
- **src**: Main model, training, data source code
- **modules**: Forked and modified source code repositories included as git submodules

### src/

Immediately relevant to the user are:

- `figures`: Pipeline to (re)produce all analyses/plots for the final manuscript (see the `src/figures/README.md` for details)
- `cellxgene_preprocessing`: Pipeline to preprocess new (single cell) RNA-seq datasets for interactive exploration in CELLxGENE/CellWhisperer

Under the hood, the main python package (`cellwhisperer`) and a series of pipelines are important:

- `cellwhisperer`: CellWhisperer embedding model python package including model, training and inference code
- `datasets`: retrieval/preparation of training and validation datasets (transcriptomes as well as annotations)
- `pre_training_processing`: Generation of natural language captions and other preparations to obtain final datasets for multimodal contrastive training
- `llava`: Pipeline for training and validation of the CellWhisperer LLM model
- `ablation`: Pipeline for embedding model ablation and evaluation
- `hosting`: Hosting infrastructure source code

### Code style

We use `blacken` for automated code formatting.

### modules/

CellWhisperer builds atop three projects that are integrated via git submodules. These were forked from original repositories on GitHub, in order to retain transparency on our code contributions as well as the option to feed back code into the upstream repository (in case of `cellxgene`).

- `llava`: CellWhisperer LLM model python package including model, training and inference code
- `cellxgene`: CELLxGENE Explorer browser package, modified to integrate UI and API elements for CellWhisperer integration
- `Geneformer`: The transcriptome model used for the CellWhisperer embedding model

<a name="analyze"/>

## Analyze your own datasets

For "latent-free" data analysis in the web browser with CellWhisperer (CELLxGENE Explorer integration), you need to preprocess your datasets. This takes from few hours up to a day or two, dependent on the dataset size and whether you have access to a GPU or a large number of CPU cores.

1. Prepare your dataset (see [guidelines below](#dataset_format_guidelines))
2. Place it in `<PROJECT_ROOT>/resources/<dataset_name>/read_count_table.h5ad`
3. Go to `cellwhisperer/src/cellxgene_preprocessing` and run the pipeline: `snakemake --use-conda --conda-frontend conda -c  <number of cores> --config 'datasets=["<dataset_name>"]'`
   - This runs much faster if you have a GPU available. If you don't have one, make sure to compensate with a substantial number of CPU cores (e.g. 32 or more)
   - Depending on your dataset, you might also require a substantial amount of RAM (e.g. your dataset size times 2)
   - We use GPT-4 or Mixtral to condense the CellWhisperer-generated cluster captions into brief titles. Set the environment variable `OPENAI_API_KEY` if you want to use the GPT-4, otherwise Mixtral is used.
4. Use the newly created file `/path/to/cellwhisperer/results/<dataset_name>/cellwhisperer_clip_v1/cellxgene.h5ad` to host a CELLxGENE Explorer instance:
  - `cellxgene launch -p 5005  --debug --host 0.0.0.0 --max-category-items 500 --var-names gene_name /path/to/cellwhisperer/results/<dataset_name>/cellwhisperer_clip_v1/cellxgene.h5ad`

### Self-host the AI models (GPU highly recommended)

For your convenience, the CellWhisperer-integrated `cellxgene` server will access our API for AI functionalities. If you instead want to run the AI models locally follow these instructions:

- for the CLIP model, simply provide the command line argument `--cellwhisperer-clip-model cellwhisperer/results/models/jointemb/cellwhisperer_clip_v1.ckpt` to the `cellxgene` command.
- for the LLM model:
  - run a controller with command `python -m llava.serve.controller --host 0.0.0.0 --port 10000` in the `llava` environment
  - run a worker with the command `python -m llava.serve.model_worker --multi-modal --host 0.0.0.0 --controller localhost:10000 --port 40000 --worker localhost:40000 --model-path /path/to/Mistral-7B-Instruct-v0.2__cellwhisperer_clip_v1/`
  - adjust the variable `CONTROLLER_URL` in `cellwhisperer/modules/cellxgene/server/common/compute/llava_utils.py` to your locally running LLM service.
  - See `hosting/home/docker-compose.yml` for an example.

For a semi-professional deployment, you may want to consider running everything within a coherent docker environment. See `hosting/home/docker-compose.yml` for a starting point.

<a name="dataset_format_guidelines"/>

### Input dataset format guidelines

We only use human data and raw read counts (not normalized) for our datasets. Normalization is taken care of by the respective transcriptome models (more specifically their processor classes) and is also performed explicitly in this preparation pipeline.

- A dataset is stored in an h5ad file
- `X` contains raw read counts and without nans (use int32)
- `var` has a *unique* index (e.g. the ensembl_id (not mandatory, but recommended)) and an additional field `gene_name` containing the gene symbol.
  - Optionally, provide an additional field "ensembl_id" (otherwise the pipeline computes it).
- If your dataset is large (i.e. > 100k cells), restrict the provided metadata fields (e.g. in `obs` and `var`) to what is really necessary
- For best results, filter cells with few expressed genes (e.g. <100 genes with expression <1)
- Try to use `categorical` instead of 'object' dtype for categorical `obs` columns
- If you want to generate cluster-labels for your own provided `obs` cluster column(s), provide a field `.uns["cluster_fields"] = ["obs_col_name1", "obs_col_name2", ...]`
- Any layouts that should make it into the webapp need to adhere to these rules:
  - stored in `.obsm` whith name `X_{name}`
  - type: `np.ndarray` (NOT `pd.DataFrame`), dtype: float/int/uint
  - shape: `(n_obs, >= 2)`
  - all values finite or NaN (NO +Inf or -Inf)
  - If you use multiple 'layers' of layouts/embeddings (e.g. sub-clustering), you can 'mask out' cells by setting them to nan
    - consider following this naming schema: `'X_umap_<name_of_obs_column>-<value_of_obs_column>` if you provide a nan-filled layout per obs-value and `'X_umap_<name_of_obs_column>` if you provide the sub-clustered embeddings in parallel in a single column.
