# Generate a MetaSRA annotated dataset from ARCHS4
This part of the repository is dedicated to the generation of a MetaSRA annotated dataset of RNA-seq samples pulled from the ARCHS4. In brief, the pipeline does the following:

1. Read the metadata from the ARCHS4 H5 file and map the contained NCBI accessions to SRA UIDs
2. Retrieve all associated accessions with these UIDs from the SRA and link them to the corresponding BioSample UIDs
3. Use BioSample UIDs to retrieve metadata from BioSample
4. Use corresponding SRA studyids to query retrieve normalized metadata from MetaSRA
5. Combine the fetched information and generate an H5AD dataset of annotated expression samples from ARCHS5 (this typically only contains the samples that have an MetaSRA annotation as this is the bottleneck here)

# How does it work
The principle workflow is outlined in the contained `Snakefile` which should work together with the `config.yaml`. The necessary environment is specified in `environment.yaml`. Additionally there are utility scripts to e.g. download ARCHS4.

# The `metadatamapping` package
The whole workflow heavily relies on the `metadatamapping` Python package developed for this usecase. And should be easily extensible to other databases.

# Entrez credentials
`metadatamapping` retrieves data from the Entrez eUtilities using the [`biopython` interface](https://biopython.org/docs/1.75/api/Bio.Entrez.html). By default the Entrez API only allows 3 requests per second if `Entrez.email` and `Entrez.api_key` are not set. This can be increased when setting these properties accordingly which also speeds up the most timeconsuming part of the pipeline which is the accession -> SRA UID mapping as this relies on eSearch which only allows for one accession at a time (maybe it also takes several but I did not test this as I expect it to be cumbersome to pull apart then). So please make sure to set the `entrez` properties in the `config.yaml` appropriately (or use mine)