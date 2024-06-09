import cellxgene_census


ALLOWED_ASSAYS = [
    "10x 3' v1",
    "10x 3' v2",
    "10x 3' v3",
    "10x 5' v1",
    "10x 5' v2",
    "10x 3' transcription profiling",
    "10x 5' transcription profiling",
    "Seq-Well",
    "Drop-seq",
    "CEL-seq2",
]
# NOTE: We therfore forbid these assays:
# ['Smart-seq v4',
#  'MARS-seq',
#  'Smart-seq2',
#  'BD Rhapsody Whole Transcriptome Analysis',
#  'GEXSCOPE technology',
#  'inDrop',
#  'microwell-seq',
#  'STRT-seq',
#  'BD Rhapsody Targeted mRNA',
#  'sci-RNA-seq']


# Get the list of datasets to process
with cellxgene_census.open_soma(census_version=CENSUS_VERSION) as census:
    cell_metadata = census["census_data"]["homo_sapiens"].obs.read(
        value_filter="is_primary_data == True", column_names=["dataset_id", "assay"]
    )
    cell_metadata = cell_metadata.concat().to_pandas()
    all_dataset_ids = set(
        cell_metadata[cell_metadata["assay"].isin(ALLOWED_ASSAYS)][
            "dataset_id"
        ].unique()
    )
