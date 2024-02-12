# single-cellm
single ceLLM project

For more information on project management, follow
https://github.com/epigen/single-cellm/wiki


## Installation instructions

1. Run `git clone git@github.com:epigen/single-cellm.git --recurse-submodules`
  If you already cloned, but did not add the `--recurse-submodules` run the following:
  `git submodule update --init --recursive`
2. Install the environment
  ```
  mamba env create -f envs/main.yaml  # this already includes `pip install -e .`
  conda activate single-cellm
  ```
3. Install scgpt and flash-attn (they need pip installation flags to be installed properly)
   ```
   bash envs/install_scgpt_after_env_creation.sh
   ```
4. Develop :)

### Installing cellxgene

See [developer_guidelines](./modules/cellxgene/dev_docs/developer_guidelines.md). In short:

- Make sure you have npm (install via conda or apt-get)
- For older versions of npm, run `export NODE_OPTIONS=--openssl-legacy-provider` ([workaround](https://stackoverflow.com/questions/69692842/error-message-error0308010cdigital-envelope-routinesunsupported))
- Build the client and put static files in place: `make build-for-server-dev`
- Install from local files: `make install-dev`
- Install prereqs for client: `make dev-env`

### Sane git defaults


1. `git config --global submodule.recurse true`
   This way you don't need to keep track of whether the submodules are up to date. Note: This may lead to code loss on the remote repo upon pull.
2. `git config --global pull.rebase true`
   Rebasing retains a better history

## Run/Train with Pytorch Lightning

We rely on pytorch lightning, which significantly reduces boilerplate for a multitude of aspects, including
- logging
- training
- CLI args
- model sharing

Before training, make sure to read the [LightningCLI documentation](https://lightning.ai/docs/pytorch/stable/cli/lightning_cli.html) (the three Basics are enough). It's short and really helpful! E.g. to start a run do this:

```bash
single_cellm fit --print_config > run_config.yaml
# configure run_config.yaml
single_cellm fit --config run_config.yaml
```

### Sweeps

To run sweeps, refer to [this README](./src/experiments/sweeps/README.md). You can run sweeps with the `single_cell_sweeping` tool

### Broken parameters

- Only AdamW optimizer works at the moment (because using it dynamically fails with --config <file>)

### Important parameters

`wandb`: Whether to log to wandb and if so which run_name to use
`trainer.logger.log_model`: Upload model to WandB?
`trainer.fast_dev_run`: Name is self-explanatory. Super useful for debugging
`ckpt_path`: a path (to load a model, e.g. for resuming)

### Run
Use `CELLWHISPERER_CACHE=/cache/cellwhisperer` to define a different cache folder

## Folder structure

- data: Computationally non-reproducible, expensive, or painful to reproduce
- metadata: Computationally non-reproducible, e.g., sample annotation sheets, clinical annotation
- results: Can be reproduced with your scripts and pipelines
- resources: External, references, datasets and tools that are project inherent and can be reproduced or downloaded with your scripts and pipelines
- src (and all other directories needed to run the source code)

### Code style

We use `blacken` for automated code formatting.

## How to install a new library (i.e. extend the environment)?

1. Load the environment, defined by envs/main.yaml with conda (`conda activate singlecellm`)
2. Install the package of interest (`conda install <your_package>`)
3. If everything works pin the package you installed in main.yaml with its version.
4. Before commiting do one of the two
4.1 Update your env conda env: `conda env update --file environment.yml`
4.2 create a fresh environment for testing from the new main.yaml (`conda env create -f envs/main.yml -n test_tmp_env`)


## Deploy
See files in `hosting/home`
