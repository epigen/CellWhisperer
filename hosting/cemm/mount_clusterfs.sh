#!/bin/sh
PATH="/home/jburton/.local/bin:/home/jburton/bin:/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin"
export PATH
# Wait up to 60 seconds before mounting
ping -c 1 -w 60 8.8.8.8
sshfs -o debug,sshfs_debug,loglevel=debug,reconnect,ServerAliveInterval=45,ServerAliveCountMax=2,IdentityFile=/home/jburton/.ssh/id_rsa jburton@login.int.cemm.at:/nobackup/lab_bock/ /nobackup/lab_bock/
