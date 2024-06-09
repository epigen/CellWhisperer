#!/bin/bash

source /opt/miniconda3/etc/profile.d/conda.sh
source activate ${CONDA_ENV}
# Need conda install versions of some libraries, e.g., to add GPU support to llama-cpp
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib"
# Execute the command provided to the docker run command
exec "$@"

# If cellxgene fails, start a shell in the conda environment
# exec /bin/bash --rcfile <(echo "source activate $CONDA_ENV")
