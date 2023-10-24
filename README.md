# single-cellm
single ceLLM project

For more information on project management, follow
https://github.com/epigen/single-cellm/wiki

## Folder structure

- data: Computationally non-reproducible, expensive, or painful to reproduce
- metadata: Computationally non-reproducible, e.g., sample annotation sheets, clinical annotation
- results: Can be reproduced with your scripts and pipelines
- resources: External, references, datasets and tools that are project inherent and can be reproduced or downloaded with your scripts and pipelines
- src (and all other directories needed to run the source code)

## How to install a new library (i.e. extend the environment)?


1. Load the environment, defined by envs/main.yaml with conda (`conda activate singlecellm`)
2. Install the package of interest (`conda install <your_package>`)
3. If everything works pin the package you installed in main.yaml with its version.
4. Before commiting do one of the two
4.1 Update your env conda env: `conda env update --file environment.yml`
4.2 create a fresh environment for testing from the new main.yaml (`conda env create -f envs/main.yml -n test_tmp_env`)

## Installation instructions

0. Make sure you have `git-lfs` installed
1. Run `git clone git@github.com:epigen/single-cellm.git --recurse-submodules`
  If you already cloned, but did not add the `--recurse-submodules` run the following:
  `git submodule update --init --recursive`
2. Install the environment
  ```
  mamba env create -f envs/main.yaml
  conda activate single-cellm
  ```

3. `pip install -e .`
4. Develop :)

### Sane git defaults


1. `git config --global submodule.recurse true`
   This way you don't need to keep track of whether the submodules are up to date
2. `git config --global pull.rebase true`
   Rebasing retains a better history




### For installing git lfs on MUW HPC, Varun did the following because we don't have sudo access:
    ```
    # 1. Create a new directory and navigate into it:
    mkdir Varun
    cd Varun
    # 2. Download git-lfs:
    wget https://github.com/git-lfs/git-lfs/releases/download/v3.4.0/git-lfs-linux-amd64-v3.4.0.tar.gz
    # 3. Verify the downloaded file:
    echo "60b7e9b9b4bca04405af58a2cd5dff3e68a5607c5bc39ee88a5256dd7a07f58c  git-lfs-linux-amd64-v3.4.0.tar.gz" | sha256sum --check
    # 4. Extract the downloaded file:
    tar -xzvf git-lfs-linux-amd64-v3.4.0.tar.gz
    # 5. Make the script executable:
    chmod +x git-lfs-3.4.0
    # 6. Add git-lfs to your PATH. Add the following to your ~/.bashrc file:
    export PATH=/msc/home/mschae83/Varun/git-lfs-3.4.0:$PATH
    # Then, apply the changes:
    source ~/.bashrc
    # 7. Install git-lfs:
    git lfs install
    # 8. Check the installed version:
    git lfs version
    ```

1. Install Git LFS. This can usually be done with the command `git lfs install`. They only need to do this once per machine.
2. Clone/Pull this repository
3. Recommend installing the environment from yaml file and installing this single-cellm as an editable module with `pip install -e .` Tested only on MUW HPC (Linux machine)...doesn't work for Mac laptop installation atm.
