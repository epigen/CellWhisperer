#!/bin/bash

source /opt/miniconda3/etc/profile.d/conda.sh
source activate ${CONDA_ENV}

# Execute the command provided to the docker run command
exec "$@"

# If cellxgene fails, start a shell in the conda environment
# exec /bin/bash --rcfile <(echo "source activate $CONDA_ENV")
