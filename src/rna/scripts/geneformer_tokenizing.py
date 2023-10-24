from geneformer import TranscriptomeTokenizer
from pathlib import Path


tk = TranscriptomeTokenizer(
    custom_attr_name_dict={
        "cell type rough": "cell type rough",
        "cell type": "cell type",
        "sample_name": "sample_name",
    }, # TODO: hardcoded metadata columns for now
    nproc=16, # TODO: hardcoded nproc for now
)
tk.tokenize_data(
    data_directory=Path(snakemake.input[0]).parent,
    output_directory=Path(snakemake.output[0]).parent,
    output_prefix=Path(snakemake.output[0]).stem,
    file_format="loom",
)
