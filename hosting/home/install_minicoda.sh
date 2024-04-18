#!/bin/bash

# Needs to be run once if mounting conda as volume (instead of as part of the docker image)

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
    $INSTALL_DIR/bin/conda init bash  # NOTE this might not be necessary
else
    echo "Miniconda is already installed in $INSTALL_DIR."
fi
