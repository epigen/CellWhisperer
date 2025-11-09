import anndata
import urllib.request
import os

# # Download the file
url = "https://datasets.cellxgene.cziscience.com/ff5a921e-6e6c-49f6-9412-ad9682d23307.h5ad"
# # Download the file
filename = os.path.basename(url)
urllib.request.urlretrieve(url, filename)

# move to output
os.rename(filename, snakemake.output.read_count_table)
