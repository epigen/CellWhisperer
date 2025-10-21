#!/bin/bash

# This script runs the daily report generator for a range of dates.
# Usage: ./backfill-reports.sh START_DATE END_DATE
# Example: ./backfill-reports.sh 2025-09-01 2025-09-30

# Validate input
if [ -z "$1" ] || [ -z "$2" ]; then
  echo "Usage: $0 START_DATE END_DATE"
  echo "Example: $0 2025-09-01 2025-09-30"
  exit 1
fi

START_DATE=$1
END_DATE=$2

# Loop through each day from start to end (inclusive)
current_date="$START_DATE"
while [ "$(date -d "$current_date" +%s)" -le "$(date -d "$END_DATE" +%s)" ]; do
  echo "----------------------------------------------------"
  echo "Backfilling report for date: $current_date"
  echo "----------------------------------------------------"
  
  # Call the main script for the specific day
  /home/jburton/cellwhisperer_private/hosting/cemm/logging/run-daily-report.sh "$current_date"
  
  # Move to the next day
  current_date=$(date -d "$current_date + 1 day" +%Y-%m-%d)
done

echo "Backfill complete."
