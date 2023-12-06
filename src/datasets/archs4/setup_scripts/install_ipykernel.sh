# run this to instantiate the conda environment and expose it to your jupyter environment
conda env create -f environment.yml
conda activate scllm
python -m ipykernel install --user --name scllm --display-name scllm
