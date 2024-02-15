# This script calls rsync to "backup" the data from meduni HPC to Lustre
# This is not a solid backup strategy, but prevents us to start "from scratch" if the MedUni HPC crashes


# Run this from Lustre:
# NOTE we don't use the -a flag, because this would also copy the group which does not match on Lustre
rsync -rtpvzh --progress mschae83@s0-l00.hpc.meduniwien.ac.at:cellwhisperer/results/ /nobackup/lab_bock/projects/cellwhisperer/results
rsync -rtpvzh --progress mschae83@s0-l00.hpc.meduniwien.ac.at:cellwhisperer/resources/ /nobackup/lab_bock/projects/cellwhisperer/resources

# Optional: also copy the repository itself (and exclude results, resources, data and metadata)
