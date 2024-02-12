#!/bin/bash

# TODO this could be split into two commands: one to install miniconda and another to start cellxgene


# Define the Miniconda version and installer filename
MINICONDA_VERSION=latest
PYTHON_VERSION=3.9

INSTALLER=Miniconda3-${MINICONDA_VERSION}-Linux-x86_64.sh
INSTALL_DIR=/opt/miniconda3

# Check if Miniconda is already installed by looking for the conda executable
if [ ! -f "$INSTALL_DIR/bin/conda" ]; then
    echo "Miniconda not found in $INSTALL_DIR. Installing..."
    # Download the Miniconda installer
    wget "https://repo.anaconda.com/miniconda/$INSTALLER" -O /tmp/miniconda.sh
    # Install Miniconda
    /bin/bash /tmp/miniconda.sh -u -b -p $INSTALL_DIR
    # Remove the installer
    rm /tmp/miniconda.sh
    # Initialize Miniconda
    $INSTALL_DIR/bin/conda init bash  # TODO this maybe needs to always be executed?
else
    echo "Miniconda is already installed in $INSTALL_DIR."
fi

source $INSTALL_DIR/etc/profile.d/conda.sh
source activate ${CONDA_ENV}
# Activate the conda environment of your choice
# source $INSTALL_DIR/bin/activate $CONDA_ENV

cellxgene launch -p ${CXG_SERVER_PORT} ${CXG_OPTIONS} ${DATASET} ${MODEL}

# If cellxgene fails, start a shell in the conda environment
exec /bin/bash --rcfile <(echo "source activate $CONDA_ENV")
