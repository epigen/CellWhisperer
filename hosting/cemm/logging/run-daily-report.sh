#!/bin/bash

# This script generates a report for a specific day.
# It defaults to "yesterday" if no date is provided as an argument.

# --- Determine the Target Date ---
if [ -z "$1" ]; then
  # Case 1: No argument provided. Default to yesterday for the cron job.
  TARGET_DATE="yesterday"
else
  # Case 2: Argument provided. Use it as the target date.
  TARGET_DATE="$1"
fi

echo "Processing logs for target date: $TARGET_DATE"

# --- Configuration ---
CONTAINER_NAME="nginx"

# --- Calculate Time Window ---
# The start of the target day (e.g., 2025-09-15T00:00:00)
START_DATE=$(date -d "$TARGET_DATE" --iso-8601=seconds)
# The start of the day *after* the target day (e.g., 2025-09-16T00:00:00)
END_DATE=$(date -d "$TARGET_DATE + 1 day" --iso-8601=seconds)

# --- Execute Pipeline ---
# 1. Get logs for the specific time window.
# 2. Pipe to the processor, passing the target date so the file is named correctly.
podman logs --since "$START_DATE" --until "$END_DATE" "$CONTAINER_NAME" 2>&1 | /home/jburton/cellwhisperer_private/hosting/cemm/logging/process_logs.sh "$TARGET_DATE"

# 3. Update the dashboard data.
python3 /home/jburton/cellwhisperer_private/hosting/cemm/logging/generate_stats.py

