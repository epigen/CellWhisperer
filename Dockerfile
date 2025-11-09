# FROM nvidia/cuda:11.6.2-base-ubuntu20.04
FROM nvidia/cuda:12.3.1-devel-ubuntu22.04

# Add some dependencies
ENV DEBIAN_FRONTEND=noninteractive

# Change the UID of the apt user to work in rootless container.
RUN sed -i 's/_apt:x:100:65534/_apt:x:100:100/g' /etc/passwd

# Explicitly pull the 55 version of the cuda-drivers, to work
# with the L4 GPUs in the hosting machine. Otherwise ubuntu
# pulls the wrong version automatically (535) with the toolkit.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git cuda-drivers-555 pkg-config libopenblas-dev

# Install any needed packages specified in requirements.txt
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    ca-certificates \
    vim \
    less \
    make \
    git \
    jq \
    nodejs \
    npm

# Note: npm installation is insanesly slow!
RUN apt-get install -y --no-install-recommends cpio

RUN rm -rf /var/lib/apt/lists/*

# Define the version of Miniconda to install
ENV MINICONDA_VERSION=py310_24.7.1-0
ENV PYTHON_VERSION=3.10
ENV CONDA_OVERRIDE_CUDA=12.5

# Define the Miniconda installer filename
ENV INSTALLER=Miniconda3-${MINICONDA_VERSION}-Linux-x86_64.sh

# Define the installation directory
ENV INSTALL_DIR=/opt/miniconda3

# Download the Miniconda installer
RUN wget --no-check-certificate "https://repo.anaconda.com/miniconda/$INSTALLER" -O /tmp/miniconda.sh

# Install Miniconda
RUN /bin/bash /tmp/miniconda.sh -b -p $INSTALL_DIR \
    && rm /tmp/miniconda.sh

# Add Miniconda to PATH
ENV PATH=$INSTALL_DIR/bin:$PATH

# Initialize Miniconda
RUN conda init --all

# Set channel_priority to flexible
RUN conda config --set channel_priority flexible

# Clone the repository (git@github.com:epigen/cellwhisperer.git --recurse-submodules) (or copy it in this case)
# RUN git clone git@github.com:epigen/cellwhisperer.git --recurse-submodules /opt/cellwhisperer
COPY --chmod=0755 . /opt/cellwhisperer

# Change the working directory
WORKDIR /opt/cellwhisperer

# Ignore SSL issues arising from proxy-ing (also for wget)
RUN npm config set strict-ssl false


# Activate the environment
COPY entrypoint.sh /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]

# Install the dependencies
RUN /entrypoint.sh bash envs/setup.sh

# Build the web app
RUN git config --global --add safe.directory /opt/cellwhisperer/modules/cellxgene
RUN git config --global --add safe.directory /opt/cellwhisperer
RUN cd modules/cellxgene && CONDA_ENV=cellwhisperer /entrypoint.sh make build-for-server-dev

# Initialize an empty git repo (the original one is ignored by .dockerignore), such that all the `git rev-parse` commands works
RUN git init

# Set the default command to run when creating a new container
CMD [ "/bin/bash" ]
