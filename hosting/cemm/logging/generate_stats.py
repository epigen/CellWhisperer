#!/usr/bin/env python3
import os
import re
import json
import glob
import datetime
from collections import defaultdict

# --- Configuration ---
REPORT_DIR = "/home/jburton/cellwhisperer_private/hosting/cemm/logging/"
JSON_OUTPUT_FILE = "/home/jburton/cellwhisperer_private/hosting/cemm/logging/dashboard/stats.json" # Or your python server path

def parse_report(filepath):
    """Extracts stats from a single daily report file."""
    stats = {'chat_sessions': 0, 'browse_only_sessions': 0}
    try:
        with open(filepath, 'r') as f:
            for line in f:
                chat_match = re.search(r"Total Unique Sessions Using Chat: (\d+)", line)
                if chat_match:
                    stats['chat_sessions'] = int(chat_match.group(1))
                browse_match = re.search(r"Browsed Main Pages but Did NOT Use Chat: (\d+)", line)
                if browse_match:
                    stats['browse_only_sessions'] = int(browse_match.group(1))
    except FileNotFoundError:
        print(f"Warning: Report file not found: {filepath}")
    return stats

def main():
    """
    Aggregates daily reports. Data from previous months is combined into single
    monthly bars. Data from the current month is shown as daily bars.
    """
    print("Generating mixed-granularity dashboard data...")

    # --- NEW: Determine the current month ---
    current_month_str = datetime.date.today().strftime("%Y-%m")

    # --- NEW: Separate dictionaries for aggregation ---
    previous_months_totals = defaultdict(lambda: {'chat_sessions': 0, 'browse_only_sessions': 0})
    current_month_daily_totals = defaultdict(lambda: {'chat_sessions': 0, 'browse_only_sessions': 0})

    report_files = sorted(glob.glob(os.path.join(REPORT_DIR, "report-user-activity-*.txt")))

    if not report_files:
        print("No report files found. Exiting.")
        return

    for report_file in report_files:
        # Extract the full YYYY-MM-DD date from the filename
        date_key_match = re.search(r'(\d{4}-\d{2}-\d{2})\.txt$', report_file)
        if not date_key_match:
            continue
        
        full_date_key = date_key_match.group(1)
        month_key = full_date_key[:7] # Extract "YYYY-MM"
        daily_stats = parse_report(report_file)

        # --- NEW: Conditional Aggregation Logic ---
        if month_key == current_month_str:
            # This report is from the current month, keep it daily
            current_month_daily_totals[full_date_key]['chat_sessions'] += daily_stats['chat_sessions']
            current_month_daily_totals[full_date_key]['browse_only_sessions'] += daily_stats['browse_only_sessions']
        else:
            # This report is from a previous month, aggregate it
            previous_months_totals[month_key]['chat_sessions'] += daily_stats['chat_sessions']
            previous_months_totals[month_key]['browse_only_sessions'] += daily_stats['browse_only_sessions']

    # --- NEW: Combine the two datasets for the chart ---
    # 1. Get sorted previous months
    sorted_previous_months = sorted(previous_months_totals.keys())
    # 2. Get sorted days of the current month
    sorted_current_days = sorted(current_month_daily_totals.keys())

    # 3. Combine them to create the final lists
    labels = sorted_previous_months + sorted_current_days
    
    chat_data = [previous_months_totals[month]['chat_sessions'] for month in sorted_previous_months] + \
                [current_month_daily_totals[day]['chat_sessions'] for day in sorted_current_days]
                
    browse_data = [previous_months_totals[month]['browse_only_sessions'] for month in sorted_previous_months] + \
                  [current_month_daily_totals[day]['browse_only_sessions'] for day in sorted_current_days]

    dashboard_data = {
        "labels": labels,
        "chatData": chat_data,
        "browseData": browse_data
    }

    with open(JSON_OUTPUT_FILE, 'w') as f:
        json.dump(dashboard_data, f, indent=2)

    print(f"Dashboard data successfully generated and written to {JSON_OUTPUT_FILE}")

if __name__ == "__main__":
    main()
