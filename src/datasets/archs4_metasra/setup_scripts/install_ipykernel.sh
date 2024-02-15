# run this to instantiate the conda environment and expose it to your jupyter environment
conda env create -f environment.yml
conda activate cellwhisperer
python -m ipykernel install --user --name cellwhisperer --display-name cellwhisperer
