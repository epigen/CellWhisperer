#!/bin/bash


CHAT_API_PATTERN="/api/v0.2/llmembs/chat"

# 2. A pattern to identify a "main page" hit.
#    This example looks for a GET request to the root URL "/".
#    The space after the slash is important to not match "/api", etc.
#    You can use a regex here, e.g., "GET /(geo|current_page|home) "
MAIN_PAGE_PATTERN='"GET /(geo|tabulasapiens|colonic_epithelium|development) '

# 3. Report location and filename.
REPORT_DIR="/home/jburton/cellwhisperer_private/hosting/cemm/logging"

if [ -z "$1" ]; then
  REPORT_DATE="yesterday"
else
  REPORT_DATE="$1"
fi
# Format the date string for the filename and title
REPORT_DATE_STR=$(date -d "$REPORT_DATE" +%Y-%m-%d)
REPORT_FILE="${REPORT_DIR}/report-user-activity-${REPORT_DATE_STR}.txt"

# --- Script Body ---
mkdir -p "$REPORT_DIR"
LOG_DATA=$(cat)

# --- Part 1: Analyze Chat API Usage (Same as before) ---
CHAT_LOGS=$(echo "$LOG_DATA" | grep "$CHAT_API_PATTERN")
CHAT_SESSION_DETAILS=""
TOTAL_CHAT_SESSIONS=0
CHAT_SESSIONS_LIST=""

if [ -n "$CHAT_LOGS" ]; then
    CHAT_SESSION_DETAILS=$(echo "$CHAT_LOGS" | awk -F'"' '{
        split($1, parts, " ");
        ip_string = parts[1];
        split(ip_string, ips, ",");
        print ips[1] " " $6;
    }' | sort | uniq -c | sort -nr)
    TOTAL_CHAT_SESSIONS=$(echo "$CHAT_SESSION_DETAILS" | wc -l)
    # Create a clean list of just the session identifiers for comparison later
    CHAT_SESSIONS_LIST=$(echo "$CHAT_SESSION_DETAILS" | awk '{$1=""; print $0}' | sed 's/^ *//')
fi

# --- Part 2: Analyze Main Page Browsers ---
MAIN_PAGE_LOGS=$(echo "$LOG_DATA" | grep -E "$MAIN_PAGE_PATTERN")
TOTAL_MAIN_PAGE_ONLY_SESSIONS=0

if [ -n "$MAIN_PAGE_LOGS" ]; then
    # Get a unique list of all sessions that hit a main page
    ALL_MAIN_PAGE_SESSIONS=$(echo "$MAIN_PAGE_LOGS" | awk -F'"' '{
        split($1, parts, " ");
        ip_string = parts[1];
        split(ip_string, ips, ",");
        print ips[1] " " $6;
    }' | sort -u)

    # Now, find the sessions that are in the main page list but NOT in the chat list
    if [ -n "$CHAT_SESSIONS_LIST" ]; then
        # Use grep to subtract the chat users from the main page users
        MAIN_PAGE_ONLY_SESSIONS=$(grep -vFf <(echo "$CHAT_SESSIONS_LIST") <(echo "$ALL_MAIN_PAGE_SESSIONS"))
    else
        # If there were no chat users, then all main page browsers are "browsers-only"
        MAIN_PAGE_ONLY_SESSIONS="$ALL_MAIN_PAGE_SESSIONS"
    fi
    TOTAL_MAIN_PAGE_ONLY_SESSIONS=$(echo "$MAIN_PAGE_ONLY_SESSIONS" | wc -l)
fi


# --- Part 3: Generate the Combined Report ---
{
  echo "Report for Date: ${REPORT_DATE_STR}"
  echo "=========================================================="
  echo ""
  echo "--- Chat API Usage ---"
  echo "[+] Total Unique Sessions Using Chat: $TOTAL_CHAT_SESSIONS"
  echo ""
  echo "[+] Hits per Chat Session (Count | IP | User Agent):"
  echo "----------------------------------------------------------"
  if [ -n "$CHAT_SESSION_DETAILS" ]; then
    echo "$CHAT_SESSION_DETAILS"
  else
    echo "No chat activity recorded."
  fi
  echo ""
  echo ""
  echo "--- Main Page Browsing (Non-Chat Users) ---"
  echo "[+] Unique Sessions that Browsed Main Pages but Did NOT Use Chat: $TOTAL_MAIN_PAGE_ONLY_SESSIONS"
  echo ""
  echo "--- End of Report ---"
} > "$REPORT_FILE"

exit 0
