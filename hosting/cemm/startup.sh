#!/bin/bash

REMOTE_USER="jburton"
REMOTE_HOST="login.int.cemm.at"
REMOTE_DIR="/nobackup/lab_bock/projects/cellwhisperer/"

# The local mount point (must exist)
MOUNT_POINT="/nobackup/lab_bock/projects/cellwhisperer/"

# A log file for debugging
LOG_FILE="/home/jburton/cellwhisperer_private/hosting/cemm/reboot.log"

echo "$(date) Reboot script started." >> "$LOG_FILE"

sleep 180

while ! ping -c 1 -W 1 8.8.8.8 >/dev/null 2>&1; do
    echo "$(date) Waiting for network." >> "$LOG_FILE"
    sleep 5
done
echo "$(date) Network is up." >> "$LOG_FILE"

if mountpoint -q "$MOUNT_POINT"; then
    echo "$(date) Filesystem is already mounted at $MOUNT_POINT." >> "$LOG_FILE"
else
    echo "$(date) Attempting to mount sshfs share..." >> "$LOG_FILE"
    sshfs -o ro,reconnect,ServerAliveInterval=45,ServerAliveCountMax=2,IdentityFile=/home/jburton/.ssh/id_rsa "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}" "$MOUNT_POINT" >> "$LOG_FILE" 2>&1

    # Check if the mount command was successful
    if [ $? -eq 0 ]; then
        echo "$(date) Mount successful." >> "$LOG_FILE"
    else
        echo "$(date) Mount FAILED. Check SSH keys and paths. Exiting." >> "$LOG_FILE"
        exit 1
    fi
fi

echo "$(date) Running post-mount commands." >> "$LOG_FILE"
# ------------------------------------------------------------------

/home/jburton/cellwhisperer_private/hosting/cemm/reset_everything.sh >> "$LOG_FILE" 2>&1
/home/jburton/cellwhisperer_private/hosting/cemm/cellwhisperer_cycle.sh >> "$LOG_FILE" 2>&1

# ------------------------------------------------------------------

echo "$(date) Reboot script finished." >> "$LOG_FILE"
