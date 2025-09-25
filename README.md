# CellWhisperer

CellWhisperer is a multimodal AI model combining transcriptomics with natural language to enable intuitive interaction with scRNA-seq datasets. The [project website](http://cellwhisperer.bocklab.org/) hosts the web tool with several example datasets as well as a short video tutorial. We also provide our model weights and curated datasets.

> This is the *private* `main` branch. Internal development is carried out here.
> The `public` branch contains the source code that is shown to the public, with full commit history.
> When releasing code to the [public repository](https://github.com/epigen/cellwhisperer), we prune the history.
> For more information on project management, follow https://github.com/epigen/cellwhisperer_private/wiki

##### Table of Contents

- [Install](#install)
- [Analyze your own datasets](#analyze)
- [Folder structure](#structure)
- [Run paper analyses](#run)

<a name="install"/>

## Install

CellWhisperer can be run through conda and docker. Both installation options run about 15 minutes. Both CPU and GPU (CUDA 12) are support.

1. Download this repository, using `git clone git@github.com:epigen/cellwhisperer.git --recurse-submodules` (*you do need the submodules*), and you can alternatively retrieve them with `git submodule update --init --recursive`)

### Install via conda/pip (recommended)

2. Install the environments

```bash
./envs/setup.sh
```

3. [Optional] Install snakemake in your `base` environment

```bash
conda install -c bioconda -n base snakemake=7
```

Alternatively, `snakemake` is accessible within the `cellwhisperer` environment (`conda activate cellwhisperer`).

4. You're good! Run the web app and analyze your datasets as described below

Note: You might need to install gcc and gxx (e.g. v9.5) if you don't have them. If you install them with conda, it might lead to [issues with snakemake](https://github.com/conda/conda/issues/6945).

### Install within Docker

You can also install and use CellWhisperer within docker, which includes all installation steps above (including cellxgene installation):

```bash
docker build -t cellwhisperer .
docker run --gpus all -it --volume .:/opt/cellwhisperer cellwhisperer bash  # also works without GPUs
conda activate cellwhisperer
```

Note that this container mounts the project directory as volume (`--volume .:/opt/cellwhisperer`) in the container (such that code modifications are visible in the container). Consider mounting also a `resources` and `results` directory to `/opt/cellwhisperer/resources` and `/opt/cellwhisperer/results`, as these are source and target directories when processing datasets (see section [Analyze your own datasets](#analyze) below).


<a name="analyze"/>


## Analyze your own dataset

To analyze your own scRNA-seq dataset with the CellWhisperer web app, follow these steps (CPU Runtime: ~2h/10,000 cells):

1. Prepare your dataset as h5ad file in `<PROJECT_ROOT>/resources/<dataset_name>/read_count_table.h5ad`
  - Make sure to adhere to the following guidelines
    - Provide raw read counts (format: int32) in `.X` or in `.layers["counts"]`
    - `.var` must have a unique index and an additional field `gene_name` containing the gene symbols.
  - Additional (optional) guidelines are provided [below](#dataset_format_guidelines)

2. Process the dataset for faster execution
  - `cd <PROJECT_ROOT>/src/cellxgene_preprocessing && snakemake --use-conda --cores 8 --config 'datasets=["<dataset_name>"]'`
  - Notes:
    - This runs much faster if you have a GPU (4GB VRAM are enough) available. Alternatively, a substantial number of CPU cores (e.g. `--cores 32`) helps as well.
    - The pipeline requires roughly two times the amount of RAM of your dataset file size.
    - We use the GPT-4 API or a locally hosted Mixtral model (GPU with 40GB VRAM recommended) to condense the CellWhisperer-generated cluster descriptions into brief captions. Provide an OpenAI API key  if you want to use GPT-4 (`export OPENAI_API_KEY=sk-abc`; Costs are negligible), otherwise Mixtral is used.
3. Run the CellWhisperer web app loading your processed dataset file
  - `cellxgene launch -p 5005  --debug --host 0.0.0.0 --max-category-items 500 --var-names gene_name <PROJECT_ROOT>/results/<dataset_name>/cellwhisperer_clip_v1/cellxgene.h5ad`

### Use AI reliably

CellWhisperer constitutes a proof-of-concept for interactive exploration of scRNA-seq data. Like other AI models, CellWhisperer does not understand user questions in a human sense, and it can make mistakes. Key results should thus be reconfirmed with conventional bioinformatics approaches.

### Self-host the AI models (GPU with 24GB VRAM highly recommended)

By default, running the web app (`cellxgene launch`) will access the API hosted at https://cellwhisperer.bocklab.org for CellWhisperer's AI capabilities. To run the AI models locally, follow these instructions:

- for the embedding model, simply provide the command line argument `--cellwhisperer-clip-model <PROJECT_ROOT>/results/models/jointemb/cellwhisperer_clip_v1.ckpt` to the `cellxgene launch` command.
- for the chat model:
  1. run a controller with command `python -m llava.serve.controller --host 0.0.0.0 --port 10000` in the `llava` environment
  2. run a worker with the command `python -m llava.serve.model_worker --multi-modal --host 0.0.0.0 --controller localhost:10000 --port 40000 --worker localhost:40000 --model-path /path/to/Mistral-7B-Instruct-v0.2__cellwhisperer_clip_v1/`
  3. adjust the variable `CONTROLLER_URL` in `cellwhisperer/modules/cellxgene/server/common/compute/llava_utils.py` to your locally running LLM service.

You may want to consider running everything within an orchestrated docker environment. See `hosting/home/docker-compose.yml` for a starting point.

<a name="dataset_format_guidelines"/>

### Input dataset format guidelines

We only support human data and raw (unnormalized) read count data for dataset processing. Normalization is performed by the respective transcriptome models (more specifically their processor classes) and is also performed explicitly in this preparation pipeline.

- A dataset is stored in an h5ad file
- Raw read counts need to be provided in `X` or in `.layers["counts"]` without nans (use int32).
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

<a name="structure"/>

## Folder structure

- **src**: Model, training, dataset and analysis source code
- **modules**: Forked and modified source code repositories included as git submodules
- results: Result files generated by analysis and training pipelines
- resources: Datasets and models downloaded by our pipelines.

### src/

Immediately relevant to the user are:

- `cellxgene_preprocessing`: Pipeline to process new (single cell) RNA-seq datasets for interactive exploration within the CELLxGENE/CellWhisperer web app
- `figures`: Pipeline to (re)produce all analyses/plots for the final manuscript (see `src/figures/README.md` for details)

Under the hood, the main python package (`cellwhisperer`) and a series of pipelines are important:

- `cellwhisperer`: CellWhisperer embedding model python package including model, training and inference code
- `datasets`: retrieval/preparation of training and validation datasets (transcriptomes as well as annotations)
- `pre_training_processing`: Generation of natural language captions and other preparations to obtain final datasets for multimodal contrastive training
- `llava`: Pipeline for training and validation of the CellWhisperer chat model
- `ablation`: Pipeline for embedding model ablation and evaluation
- `hosting`: Hosting infrastructure source code

### Code style

We use `blacken` for automated code formatting.

### modules/

CellWhisperer builds atop three projects that are integrated via git submodules. These were forked from original repositories on GitHub, in order to retain transparency on our code contributions.

- `llava`: CellWhisperer chat model python package including model, training and inference code
- `cellxgene`: CELLxGENE Explorer browser package, modified to integrate UI and API elements for CellWhisperer integration
- `Geneformer`: The transcriptome model used for the CellWhisperer embedding model


<a name="run"/>

## Run paper analyses

All training data, evaluation data and model weights are downloaded automatically, and can be browsed on [our file server](http://medical-epigenomics.org/papers/schaefer2025cellwhisperer/)

### Reproduce manuscript analyses and plots

We provide all our validations and analyses in a single pipeline, (re)producing all (*) plots in our paper.

Note that due to the high computational cost, this pipeline relies on some precomputed files, which are downloaded from our server as part of the pipeline. Nevertheless computing all the analyses will require a considerable amount of storage (~1TB), RAM (up to 1TB), GPU (40GB VRAM) and time (2 days) resources. You will need a huggingface token to download the "mistral" and "llama-3.3" models (needed for Figure 4 evaluations)

To run the pipeline, execute

```bash
cd src
snakemake --use-conda -k  # optionally only download "models" or generate "figures"
```

(*) Some interactive analyses/screenshots were performed directly in the CELLxGENE CellWhisperer browser integration and are not reproduced by the pipeline. Also note that some analyses are skipped by downloading intermediary results files due to extensive resource requirements or to prevent the need for an OpenAI API key.

Refer to `src/figures/README.md` for further details.

### Training data curation for CellWhisperer training

This is a resource-intensive endeavor and not fully automated. We provide the full processed datasets for your convenience (https://medical-epigenomics.org/papers/schaefer2025cellwhisperer/data/datasets/archs4_geo/full_data.h5ad and https://medical-epigenomics.org/papers/schaefer2025cellwhisperer/data/datasets/cellxgene_census/full_data.h5ad) with the curated natural language captions in `.obs["natural_language_annotation"]`.

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

We rely on pytorch lightning (for a primer, read [LightningCLI documentation](https://lightning.ai/docs/pytorch/stable/cli/lightning_cli.html); the three 'Basic' tutorials are a good start). To start a run, execute this:

```bash
cd cellwhisperer  # go to the project directory to line up all paths correctly
cellwhisperer fit --print_config > run_config.yaml
# Edit the run_config.yaml
cellwhisperer fit --config run_config.yaml
```

Our config for training CellWhisperer is located at `src/cellwhisperer_clip_v1.yaml`

#### Important parameters

- `wandb`: Provide a name if you want to log to wandb
- `trainer.fast_dev_run`: Useful for debugging
- `ckpt_path`: a checkpoint path to load a model for resuming training)

### Train CellWhisperer chat model

1. Go to `src/llava`
2. Run `snakemake`

Note 1: The pipelines includes code to generate the datasets. Since this takes a considerable amount of time and computational resources, we recommend downloading our provided data set. (automatically done by the snakemake pipeline defined in `src/Snakefile`).
Note 2: You might be requested to login to huggingface to be able to download the Mistral-7B model. Simply follow the instructions printed in the command line. The `huggingface-cli` tool is installed in the `cellwhisperer` environment.

### SLURM clusters and snakemake

While the easiest way to run `snakemake` is on a local or allocated machine, you can also use it for automated job deployment on HPC clusters such as SLURM. Follow [the snakemake docs](https://snakemake.readthedocs.io/en/v7.7.0/executing/cluster.html) to set up a config profile for your cluster. You'll likely need to modify the `slurm_gres` function defined in `src/shared/config.smk` to reflect your cluster's resource identifiers.
