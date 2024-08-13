# Cycles the CellWhisperer containers on CeMM infrastructure.
# Due to apparent issues with podman-compose setting up and tearing down networks
# leading to dns issues we manage the network lifecycle externally.
# 
# That is, after repeated docker up/downs containers start to crash because they can't
# resolve one another by hostname. Keeping the container environment clean by stopping
# all the containers and cleaning all the dangling resources with prune seems to 
# prevent this issue.

# First stop any running cell whisperer containers
docker compose down
# Clean up any dangling resources (containers, networks)
docker container prune -f
podman network prune -f

# Create a new network for this cycle of cellwhisperer.
export PODMAN_NETWORK_NAME=$(tr -dc A-Za-z0-9 </dev/urandom | head -c 10;)
podman network create ${PODMAN_NETWORK_NAME}

# Start the cellwhisperer containers.
docker compose up -d --remove-orphans
