# CellWhisperer

CellWhisperer project

TODO For more information on project management, follow https://github.com/epigen/cellwhisperer/wiki

## Install

### Install via conda/pip

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

3. Install the environments
  ```
  mamba env create -f envs/main.yaml  # name: cellwhisperer   this already includes `pip install -e .`
  mamba env create -f envs/llava.yaml  # name: llava    this already includes `pip install -e ../modules/LLaVA/[train]`

  conda activate cellwhisperer
  ```
4. Develop :)

### Install within Docker

You can also install and use CellWhisperer within docker:

```bash
docker build -t cellwhisperer .
docker run --gpus all -it --volume .:/opt/cellwhisperer cellwhisperer bash
conda activate cellwhisperer
```

Note that this container loads the project directory as volume (`--volume .:/opt/cellwhisperer`) in the container (such that modifications are visible in the container).

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

### Reproduce manuscript analyses and plots

We provide all our validations and analyses in a single pipeline, (re)producing all (*) plots in our paper.

Note that due to the high computational cost, this pipeline relies on some precomputed files, which are downloaded from our server as part of the pipeline. Nevertheless computing all the analyses will require a considerable amount of storage (~1TB), RAM (~1TB), CPU (~100 cores), GPU and time (2 days) resources.

To run the pipeline, execute

```bash
cd src
snakemake --use-conda
```

(*) Some interactive analyses/screenshots were performed directly in the CELLxGENE CellWhisperer browser integration and are not reproduced by the pipeline. Also note that some analyses (e.g. Extended Data Figure 3) depend on GPT-4 and the availability of an OpenAI API key.

Refer to `src/figures/README.md` for further details.

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

The config used to train is located at `src/cellwhisperer_clip_v1.yaml`

#### Important parameters

`wandb`: Whether to log to wandb and if so which run_name to use
`trainer.logger.log_model`: Upload model to WandB?
`trainer.fast_dev_run`: Name is self-explanatory. Super useful for debugging
`ckpt_path`: a path (to load a model, e.g. for resuming)

#### TODO Sweeps

To run sweeps, refer to [this README](./src/experiments/sweeps/README.md). You can run sweeps with the `single_cell_sweeping` tool

### Train CellWhisperer LLM

1. Go to `src/llava`
2. Run `snakemake`

Note: the pipelines includes code to generate the datasets. Since this takes a considerable amount of time and computational resources, we recommend downloading our provided data set. (automatically done by the main pipeline in `src`).

## Folder structure

- data: Computationally non-reproducible, expensive, or painful to reproduce
- results: Can be reproduced with your scripts and pipelines
- resources: External, references, datasets and tools that are project inherent and can be reproduced or downloaded with your scripts and pipelines
- **src**: Main model, training, data source code
- **modules**: Forked and modified source code repositories

### src/

Immediately relevant to the user are:

- `figures`: Pipeline to (re)produce all analyses/plots for the final manuscript (see the `src/figures/README.md` for details)
- `cellxgene_preprocessing`: Pipeline to preprocess new (single cell) RNA-seq datasets for interactive exploration in CELLxGENE/CellWhisperer

These modules and pipelines are used 'under the hood':

- `cellwhisperer`: CellWhisperer embedding model python package including model, training and inference code
- `datasets`: retrieval/preparation of training and validation datasets (transcriptomes as well as annotations)
- `pre_training_processing`: Generation of natural language captions and other preparations to obtain final datasets for multimodal contrastive training
- `llava`: Pipeline for training and validation of the CellWhisperer LLM model
- `ablation`: Pipeline for embedding model ablation and evaluation
- `hosting`: Hosting infrastructure source code

### Code style

We use `blacken` for automated code formatting.

### modules/

CellWhisperer builds atop two projects that are integrated via git submodules. These were forked from original repositories on GitHub, in order to retain transparency on our code contributions as well as the option to feed back code into the upstream repository (in case of `cellxgene`).

- `llava`: CellWhisperer LLM model python package including model, training and inference code
- `cellxgene`: CELLxGENE Explorer browser package, modified to integrate UI and API elements for CellWhisperer integration
- `Geneformer`: The transcriptome model used for the CellWhisperer embedding model

## Processing of (single cell) datasets and use within CELLxGENE

For an efficient use of CellWhisperer in the web browser (CELLxGENE Explorer integration), you need to preprocess your datasets.

1. Prepare your dataset (for guidelines see below)
2. Place it in `<PROJECT_ROOT>/resources/<dataset_name>/read_count_table.h5ad`
3. Go to `cellwhisperer/src/cellxgene_preprocessing` and run the pipeline: `snakemake --config 'datasets=["<dataset_name>"]'`
   - This runs much faster if you use a GPU. Also, depending on your dataset, this might require a substantial amount of RAM.
   - We use GPT-4 or Mixtral to condense the CellWhisperer-generated cluster captions into brief titles. Set the environment variable `OPENAI_API_KEY` if you want to use the GPT-4, otherwise Mixtral is used.
4. Use the newly created file `snakemake /path/to/cellwhisperer/results/<dataset_name>/cellwhisperer_clip_v1/cellxgene.h5ad` to host a CELLxGENE Explorer instance:
   - `cellxgene launch -p 5005  --debug --host 0.0.0.0 --max-category-items 500 --var-names gene_name /path/to/cellwhisperer/results/<dataset_name>/cellwhisperer_clip_v1/cellxgene.h5ad /path/to/cellwhisperer/results/models/jointemb/cellwhisperer_clip_v1.ckpt`
   - For a docker-driven deployment refer to `hosting/home`
   - NOTE The CellWhisperer LLM integration relies on an additionally running job (see `hosting/home/docker-compose.yml`). TODO: Add instructions on how to provide the LLM service here

### Dataset input format guidelines

We only use human data and raw read counts (not normalized) for our datasets. Normalization is taken care of by the respective transcriptome models (more specifically their processor classes) and is also performed explicitly in this preparation pipeline.

- A dataset is stored in an h5ad file
- `X` contains raw read counts and without nans (use int32)
- `var` has a *unique* index (e.g. the ensembl_id (not mandatory, but recommended)) and an additional field `gene_name` containing the gene symbol.
  - Optionally, provide an additional field "ensembl_id" (otherwise the pipeline computes it).
- If your dataset is large (i.e. > 100k cells), restrict the provided metadata fields (e.g. in `obs` and `var`) to what is really necessary
- For best results, filter cells with few expressed genes (e.g. <100 genes with expression <1)
- Try to use `categorical` instead of 'object' dtype for categorical `obs` columns
- Any layouts that should make it into the webapp need to adhere to these rules:
  - stored in `.obsm` whith name `X_{name}`
  - type: `np.ndarray` (NOT `pd.DataFrame`), dtype: float/int/uint
  - shape: `(n_obs, >= 2)`
  - all values finite or NaN (NO +Inf or -Inf)
