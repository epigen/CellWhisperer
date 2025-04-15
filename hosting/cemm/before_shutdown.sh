#!/bin/bash

#https://forum.manjaro.org/t/running-a-systemd-user-service-before-shutdown/143440/3
cd "$(dirname "$0")"
echo Logout service triggered on `date` >> /home/jburton/cellwhisperer_private/hosting/cemm/before_shutdown.log
docker compose down 2>&1 >> /home/jburton/cellwhisperer_private/hosting/cemm/before_shutdown.log
# First stop any running cell whisperer containers
podman pod stop --all 2>&1 >> /home/jburton/cellwhisperer_private/hosting/cemm/before_shutdown.log
podman stop --all 2>&1 >> /home/jburton/cellwhisperer_private/hosting/cemm/before_shutdown.log
# Clean up any dangling resources (containers, networks)
docker container prune -f 2>&1 >> /home/jburton/cellwhisperer_private/hosting/cemm/before_shutdown.log
podman network prune -f 2>&1 >> /home/jburton/cellwhisperer_private/hosting/cemm/before_shutdown.log
sleep 3
echo Logout service ended on `date` >> /home/jburton/cellwhisperer_private/hosting/cemm/before_shutdown.log
