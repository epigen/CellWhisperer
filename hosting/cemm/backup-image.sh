#!/bin/bash
printf -v date '%(%Y-%m-%d)T' -1
line=$(podman images | grep cellwhisperer)
read _ _ hash _ <<< $line
podman save -o /nobackup/lab_bock/projects/cellwhisperer/images/$date-cellwhisperer-image-$hash.tar $hash
