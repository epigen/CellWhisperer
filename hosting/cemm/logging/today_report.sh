#!/bin/bash

CONTAINER_NAME="nginx"

podman logs --since 24h "$CONTAINER_NAME" 2>&1 | /home/jburton/cellwhisperer_private/hosting/cemm/logging/process_logs.sh