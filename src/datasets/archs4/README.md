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