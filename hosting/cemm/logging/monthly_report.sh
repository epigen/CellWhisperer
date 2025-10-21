#!/bin/bash

CONTAINER_NAME="nginx"

# --since: First moment of the previous month (e.g., 2025-09-01T00:00:00)
# --until: First moment of the current month (e.g., 2025-10-01T00:00:00)
START_DATE=$(date -d "$(date +%Y-%m-01) -1 month" --iso-8601=seconds)
END_DATE=$(date -d "$(date +%Y-%m-01)" --iso-8601=seconds)

podman logs --since "$START_DATE" --until "$END_DATE" "$CONTAINER_NAME" | /home/jburton/cellwhisperer_private/hosting/cemm/logging/process_logs.sh