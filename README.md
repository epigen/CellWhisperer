# CellWhisperer

CellWhisperer is a multimodal AI model combining transcriptomics with natural language to enable intuitive interaction with scRNA-seq datasets. CellWhisperer is [published in Nature Biotechnology](https://doi.org/10.1038/s41587-025-02857-9). The [project website](http://cellwhisperer.bocklab.org/) hosts the web tool with several example datasets as well as a short video tutorial. We also provide our model weights and curated datasets.

This repository contains detailed instructions on how to run your own CellWhisperer instance and import custom datasets, as well as the full source code, models, and training data.

##### Table of Contents

- [Installation](#install)
- [Analyze Your Own Datasets](#analyze)
- [Folder Structure](#structure)
- [Reproducing Paper Analyses](#run)
- [Citation and Contact](#citation)

<a name="install"/>

## Installation

Installing a local copy of CellWhisperer allows you to analyze your own datasets and explore scRNA-seq data interactively using the CellWhisperer AI model. The installation process takes approximately 15 minutes and supports both CPU and GPU (CUDA 12) environments.

1. Download this repository, using `git clone https://github.com/epigen/cellwhisperer.git --recurse-submodules` (*you do need the submodules*), and you can alternatively retrieve them with `git submodule update --init --recursive`)

### Installation Steps

1. **Clone the repository** with all submodules (required):
   ```bash
   git clone git@github.com:epigen/cellwhisperer.git --recurse-submodules
   cd cellwhisperer
   ```
   
   If you've already cloned without submodules, retrieve them with:
   ```bash
   git submodule update --init --recursive
   ```

2. **Set up the conda environments:**
   ```bash
   ./envs/setup.sh
   ```
   
   This script creates the necessary conda environments including `cellwhisperer` (main environment) and `llava` (for the chat model).

3. **Install snakemake** (optional, for running paper analyses):
   ```bash
   conda install -c bioconda -n base snakemake=7
   ```
   
   Alternatively, `snakemake` is accessible within the `cellwhisperer` environment after activation.

4. **Verify installation:**
   Activate the environment and check that cellxgene is available:
   ```bash
   conda activate cellwhisperer
   cellxgene --version
   ```

**Note on compilers:** If you encounter build issues, you may need to install gcc and g++ (version 9.5 recommended). If installing via conda, be aware of potential [compatibility issues with snakemake](https://github.com/conda/conda/issues/6945).

You're now ready to run CellWhisperer locally (see next section) or analyze your own datasets.

### Alternative: Docker Installation

For users who prefer containerized environments, CellWhisperer can be installed and run using Docker. This approach includes all dependencies and installation steps in a self-contained environment.

1. **Build the Docker image:**
   ```bash
   docker build -t cellwhisperer .
   ```

2. **Run the container:**
   ```bash
   docker run --gpus all -it --volume .:/opt/cellwhisperer cellwhisperer bash
   # Also works without GPUs (omit --gpus all)
   ```

3. **Activate the environment inside the container:**
   ```bash
   conda activate cellwhisperer
   ```

**Note on volumes:** The command above mounts the project directory as a volume (`--volume .:/opt/cellwhisperer`) so that code modifications are visible inside the container. For processing datasets, consider also mounting `resources` and `results` directories:

```bash
docker run --gpus all -it \
  --volume .:/opt/cellwhisperer \
  --volume /path/to/resources:/opt/cellwhisperer/resources \
  --volume /path/to/results:/opt/cellwhisperer/results \
  cellwhisperer bash
```

<a name="analyze"/>

## Analyze Your Own Datasets

CellWhisperer can analyze your own scRNA-seq datasets through a straightforward three-step process. We currently support human data with raw (unnormalized) read counts.

**Processing time:** Approximately 2 hours per 10,000 cells on CPU (significantly faster with GPU).

### Step 1: Prepare Your Dataset

Place your dataset as h5ad file at `<PROJECT_ROOT>/resources/<dataset_name>/read_count_table.h5ad` with the following requirements:

**Required:**
- Raw read counts (int32 format) in `.X` or `.layers["counts"]`
- `.var` must have a unique index (e.g., Ensembl IDs) and a `gene_name` field with gene symbols
- No NaN values in the count matrix

**Recommended:**
- Filter cells with few expressed genes (e.g., <100 genes with counts >1)
- Use `categorical` dtype for categorical columns in `.obs`
- Provide an `ensembl_id` field in `.var` (will be computed if missing)
- For large datasets (>100k cells), keep only essential metadata fields

See [Input Dataset Format Guidelines](#dataset_format_guidelines) below for more details.

### Step 2: Process the Dataset

Run the preprocessing pipeline to generate embeddings and prepare the dataset for CellWhisperer:

```bash
cd <PROJECT_ROOT>/src/cellxgene_preprocessing
snakemake --use-conda --cores 8 --config 'datasets=["<dataset_name>"]'
```

**Important notes:**
- **GPU acceleration:** Processing is considerably faster with a GPU (4GB VRAM sufficient). Without GPU, increase CPU cores (e.g., `--cores 32`). To specify which GPU to use, set the `CUDA_VISIBLE_DEVICES` environment variable (e.g., `export CUDA_VISIBLE_DEVICES=0` for the first GPU).
- **Memory requirements:** Allow approximately 2× the dataset file size in RAM.
- **Cluster captions:** The pipeline uses GPT-4 API or a locally hosted Mixtral model to summarize CellWhisperer descriptions into brief cluster captions. To use GPT-4 (recommended, cost is low), set: `export OPENAI_API_KEY=sk-your-key`. Otherwise, Mixtral will be used (requires GPU with 40GB VRAM).

### Step 3: Launch CellWhisperer

Start the web interface with your processed dataset:

```bash
conda activate cellwhisperer
cellxgene launch -p 5005 --host 0.0.0.0 --max-category-items 500 \
  --var-names gene_name \
  <PROJECT_ROOT>/results/<dataset_name>/cellwhisperer_clip_v1/cellxgene.h5ad
```

Access the interface at `http://localhost:5005` and start exploring your data with natural language queries! (If port 5005 is already in use, you can change it by modifying the `-p` parameter to any available port.)

### Optional: Self-host the AI models

By default, the web app accesses the CellWhisperer API hosted at https://cellwhisperer.bocklab.org for interactive AI capabilities (i.e. the chat interface and the generation of CellWhisperer scores for given queries; cell embeddings and cluster descriptions are generated locally during Step 2). This setup allows you to run CellWhisperer smoothly without local GPU resources for the web interface.

If you prefer to run the AI models for the web interface locally:

1. **For the embedding model** (requires 4GB VRAM), add the following argument to the `cellxgene launch` command:
   ```bash
   --cellwhisperer-clip-model <PROJECT_ROOT>/results/models/jointemb/cellwhisperer_clip_v1.ckpt
   ```

2. **For the chat model** (requires 20GB VRAM), you need to run separate services:
   
   In one terminal (controller):
   ```bash
   conda activate llava
   python -m llava.serve.controller --host 0.0.0.0 --port 10000
   ```
   
   In another terminal (model worker):
   ```bash
   conda activate llava
   python -m llava.serve.model_worker --multi-modal --host 0.0.0.0 \
     --controller localhost:10000 --port 40000 --worker localhost:40000 \
     --model-path <path_to_mistral_model>
   ```
   
   Then adjust the `WORKER_URL` variable in `modules/cellxgene/server/common/compute/llava_utils.py` to point to your local controller.

### Important: Use AI Cautiously

CellWhisperer constitutes a proof-of-concept for interactive exploration of scRNA-seq data. Like other AI models, CellWhisperer does not understand user questions in a human sense, and it can make mistakes. **Key results should always be reconfirmed with conventional bioinformatics approaches.**

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
- Any 2D visualizations/embeddings (e.g., UMAP, t-SNE) that should be available in the webapp need to adhere to these rules:
  - stored in `.obsm` with name `X_{name}`
  - type: `np.ndarray` (NOT `pd.DataFrame`), dtype: float/int/uint
  - shape: `(n_obs, >= 2)`
  - all values finite or NaN (NO +Inf or -Inf)
  - If you use multiple 'layers' of layouts/embeddings (e.g. sub-clustering), you can 'mask out' cells by setting them to nan
    - consider following this naming schema: `'X_umap_<name_of_obs_column>-<value_of_obs_column>` if you provide a nan-filled layout per obs-value and `'X_umap_<name_of_obs_column>` if you provide the sub-clustered embeddings in parallel in a single column.

<a name="structure"/>

## Folder Structure

This section provides an overview of the repository organization to help you navigate the codebase.

```
cellwhisperer/
├── src/               # Source code for models, training, and analyses
├── modules/           # Git submodules for modified external dependencies
├── results/           # Generated results from pipelines (created during use)
├── resources/         # Downloaded datasets and models (created during use)
└── envs/              # Conda environment configurations
```

### src/ Directory

To analyze your own data or to reproduce analyses from our paper, these directories are most relevant:

- **`cellxgene_preprocessing/`**: Pipeline to process new scRNA-seq datasets for use with CellWhisperer
- **`figures/`**: Pipeline to reproduce all analyses and plots from the paper (see `src/figures/README.md` for details)

**For developers and researchers, these contain the core implementation:**

- **`cellwhisperer/`**: Main Python package with the embedding model, training code, and inference utilities
- **`datasets/`**: Scripts for retrieving and preparing training/validation datasets (transcriptomes and annotations)
- **`pre_training_processing/`**: Natural language caption generation and dataset preparation for contrastive training
- **`llava/`**: Training and validation pipeline for the CellWhisperer chat model
- **`ablation/`**: Ablation studies and evaluation pipelines for the embedding model

### modules/ Directory (Git Submodules)

CellWhisperer builds upon three external projects, integrated as git submodules. These were forked from their original repositories to maintain transparency regarding our modifications:

- **`llava/`**: Chat model implementation (modified LLaVA architecture)
- **`cellxgene/`**: Modified CELLxGENE Explorer with CellWhisperer UI and API integration
- **`Geneformer/`**: Transcriptome foundation model used in CellWhisperer's embedding architecture

### Code Style

We use `blacken` for automated Python code formatting. To format code:

```bash
conda activate cellwhisperer
blacken <file_or_directory>
```


<a name="run"/>

## Reproducing Paper Analyses

All training data, evaluation data and model weights are downloaded automatically, and can be browsed on [our file server](http://medical-epigenomics.org/papers/schaefer2025cellwhisperer/)

### Reproduce manuscript analyses and plots

We provide all our validations and analyses in a single pipeline, (re)producing all (*) plots in our paper.

Note that due to the high computational cost, this pipeline relies on some precomputed files, which are downloaded from our server as part of the pipeline. Nevertheless computing all the analyses will require a considerable amount of storage (~1TB), RAM (up to 1TB), GPU (40GB VRAM) and time (approximately 1 week) resources. You will need a huggingface token to download the "mistral" and "llama-3.3" models (needed for Figure 4 evaluations)

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

### SLURM Clusters and Snakemake

While the easiest way to run `snakemake` is on a local or allocated machine, you can also use it for automated job deployment on HPC clusters such as SLURM. Follow [the snakemake docs](https://snakemake.readthedocs.io/en/v7.7.0/executing/cluster.html) to set up a config profile for your cluster. You'll likely need to modify the `slurm_gres` function defined in `src/shared/config.smk` to reflect your cluster's resource identifiers.

---

<a name="citation"/>

## Citation and Contact

If you use CellWhisperer in your research, please cite our paper:

**Moritz Schaefer\*, Peter Peneder\*, Daniel Malzl, Salvo Danilo Lombardo, Mihaela Peycheva, Jake Burton, Anna Hakobyan, Varun Sharma, Thomas Krausgruber, Celine Sin, Jörg Menche, Eleni M. Tomazou, Christoph Bock.** *Multimodal learning enables chat-based exploration of single-cell data.* Nature Biotechnology, https://doi.org/10.1038/s41587-025-02857-9

### Questions or Feedback?

For questions or additional information, please contact the authors of the paper:
- Email: cellwhisperer@bocklab.org
- GitHub Issues: https://github.com/epigen/CellWhisperer/issues

We welcome feedback and contributions to improve CellWhisperer!
