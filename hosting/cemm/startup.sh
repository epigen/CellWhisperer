#!/bin/sh
set -x
set -e
PATH="/home/jburton/.local/bin:/home/jburton/bin:/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin"
export PATH
# Wait for network before  mounting
while ! ping -c 1 -w 60 8.8.8.8 &> /dev/null
do
	sleep 1
done
mkdir -p /nobackup/lab_bock/projects/cellwhisperer
sleep 5

while ! pgrep -f "sshfs.*/nobackup/lab_bock/projects/cellwhisperer/" > /dev/null; do
    echo "Trying to start sshfs"
    sshfs -o ro,debug,sshfs_debug,loglevel=debug,reconnect,ServerAliveInterval=45,ServerAliveCountMax=2,IdentityFile=/home/jburton/.ssh/id_rsa jburton@login.int.cemm.at:/nobackup/lab_bock/projects/cellwhisperer/ /nobackup/lab_bock/projects/cellwhisperer/ 1> /dev/null 2> /dev/null &
    sleep 2
done
echo "SSHFS started"

/home/jburton/cellwhisperer_private/hosting/cemm/cellwhisperer_cycle.sh

wait
