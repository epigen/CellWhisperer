#!/bin/bash
#
#SBATCH --job-name=metasra
#SBATCH --cpus-per-task=1
#SBATCH --mem=8GB
#SBATCH --array=1-289
#SBATCH --output=out/metasra_%a.out
#SBATCH --mail-type=ALL
#SBATCH --qos=medium
#SBATCH --time=01-12:00:00
#SBATCH --mail-user=human_disease.malzl@imp.ac.at

input_json=metadata/raw_biosample_metadata_${SLURM_ARRAY_TASK_ID}.json
output_json=metadata/normalized_biosample_metadata_${SLURM_ARRAY_TASK_ID}.json

echo $input_json
echo $output_json

./scripts/run_metasra.sh $input_json $output_json
