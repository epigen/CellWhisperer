# About the Snakefile
The Snakefile is a central resource of the code executed to arrive at the final output (the archs4 dataset). However, I never executed it using Snakemake but rather did it via the provided notebooks. Thus, the Snakefile is a condensed and ordered collection of code necessary for producing the output but might not work because I don't know how to build pipelines in Snakemake

More information below.


# Generate a MetaSRA annotated dataset from ARCHS4
This part of the repository is dedicated to the generation of a MetaSRA annotated dataset of RNA-seq samples pulled from the ARCHS4. In brief, the pipeline does the following:

1. Read the metadata from the ARCHS4 H5 file and map the contained NCBI accessions to SRA UIDs
2. Retrieve all associated accessions with these UIDs from the SRA and link them to the corresponding BioSample UIDs
3. Use BioSample UIDs to retrieve metadata from BioSample
4. Use corresponding SRA studyids to query retrieve normalized metadata from MetaSRA
5. Combine the fetched information and generate an H5AD dataset of annotated expression samples from ARCHS5 (this typically only contains the samples that have an MetaSRA annotation as this is the bottleneck here)

# How does it work
The principle workflow is outlined in the contained `Snakefile` which should work together with the `config.yaml`. The necessary environments are specified in `envs` directory (`envs/metadatamapping.yaml` is needed to run most of the metadata fetching `envs/metasra.yaml` is needed to run the MetaSRA pipeline). Additionally there are utility scripts to e.g. download ARCHS4.
Before you can run it make sure to check below and set up `metasra` properly.
Furthermore, the as I am not proficient in SnakeMake the Snakefile might (definitely) be not working and only contains the principle flow of code execution. So please make sure to fix it or use the notebooks.

# The `metadatamapping` package
The whole workflow heavily relies on the [`metadatamapping`](https://github.com/dmalzl/metadatamapping) Python package developed for this usecase and should be easily extensible to other databases.

# Entrez credentials
`metadatamapping` retrieves data from the Entrez eUtilities using the [`biopython` interface](https://biopython.org/docs/1.75/api/Bio.Entrez.html). By default the Entrez API only allows 3 requests per second if `Entrez.email` and `Entrez.api_key` are not set. This can be increased when setting these properties accordingly which also speeds up the most timeconsuming part of the pipeline which is the accession -> SRA UID mapping as this relies on eSearch which only allows for one accession at a time (maybe it also takes several but I did not test this as I expect it to be cumbersome to pull apart then). So please make sure to set the `entrez` properties in the `config.yaml` appropriately (or use mine)

# Running MetaSRA locally or retrieving the normalized metadata from the API
MetaSRA normalized metadata can be retrieved in two ways (i) run the MetaSRA pipeline locally on you samples or (ii) retrieve the normalized metadata from the MetaSRA API. While the latter involves less overhead and might also be faster the current version (1.8) of the MetaSRA database only yielded an overlap of 162,566 samples with ARCHS4 which is ~1/4 of the full dataset. Thus if you want to make sure that all your samples are retained you need to run MetaSRA locally as described below and also detailed in the `Snakefile`. Comparing the API retrieved metadata to the locally generated ones there are only 196 of the 162,566 samples that have diverging annotations where 186 of those samples did not have an API annotation but were annotated by the local run. The remaining 10 did have an API annotation but failed to be annotated locally. This result might be a consequence of updated ontologies and changes to the pipeline code since the MetaSRA database was last published but all in all we can be confident that local ~ API.

# About running MetaSRA
The original MetaSRA pipeline is unusable as is so I forked it and added the fixed fork as a submodule. You should be able to run with `scripts/run_metasra.sh` it after installing with `setup_scripts/install_metasra.sh` like so 
```bash
./scripts/install_metasra.sh
./scripts/run_metasra.sh <sra_metadata.json>
```
please see `test_data/test_in.json` or [MetaSRA](https://github.com/dmalzl/metasra/tree/master) for more infos on the format of `sra_metadata.json`.
Since the mapping is quite slow (an estimated 48h for 10k samples) we run it in parallel using SLURM, which can easily be reproduced with the `convert_biosample_metadata_to_json.ipynb` notebook and then running `run_batched_metasra.sh`

