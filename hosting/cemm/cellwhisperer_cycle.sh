#!/bin/bash
# Cycles the CellWhisperer containers on CeMM infrastructure.
# Due to apparent issues with podman-compose setting up and tearing down networks
# leading to dns issues we manage the network lifecycle externally.
# 
# That is, after repeated docker up/downs containers start to crash because they can't
# resolve one another by hostname. Keeping the container environment clean by stopping
# all the containers and cleaning all the dangling resources with prune seems to 
# prevent this issue.

# Jump to the directory that contains the script (also contains docker-compose.yaml)
cd "$(dirname "$0")"

# Make sure that the network filesystem is mounted
if ! timeout 100 bash -c 'until [ -d /nobackup/lab_bock/projects/cellwhisperer/ ]; do sleep 1; done'; then
  echo "Error: Directory /nobackup/lab_bock/projects/cellwhisperer/ not found after 100 seconds."
  exit 1
fi


# First stop any running cell whisperer containers
docker compose down
# Clean up any dangling resources (containers, networks)
docker container prune -f
podman network prune -f

# Create a new network for this cycle of cellwhisperer.
export PODMAN_NETWORK_NAME=$(tr -dc A-Za-z0-9 </dev/urandom | head -c 10;)
podman network create --subnet 10.89.1.1/24 ${PODMAN_NETWORK_NAME} # --ipam-driver="dhcp"

# Start the cellwhisperer containers.
docker compose up -d --remove-orphans
#podman-compose --in-pod cellwhisperer_group --pod-args='--infra=true' up -d
