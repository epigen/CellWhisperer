rule prepare_gsva_dataset:
    """
    TODO might need to use layers["counts"]
    """
    input:
        read_count_table=PROJECT_DIR / config["paths"]["read_count_table"],
        weights=rules.local_density_to_sample_weight.output.weight.format(dataset="{dataset}", modality="transcriptome")  # could also take mean of transcriptome and annotation
    output:
        sparse_matrix=PROJECT_DIR / "results" / "{dataset}" / "tmp" / "gsva_prepared.mtx",
        colnames=PROJECT_DIR / "results" / "{dataset}" / "tmp" / "gsva_prepared.columns",
        rownames=PROJECT_DIR / "results" / "{dataset}" / "tmp" / "gsva_prepared.rows"
    params:
        filter_protein_coding=lambda wildcards: wildcards.dataset == "archs4_geo",
        seed=42,
        num_samples=50000
    conda:
        "cellwhisperer"
    resources:
        mem_mb=500000,
        slurm="cpus-per-task=2"
    notebook:
        "../notebooks/prepare_gsva_dataset.py.ipynb"


rule gsva:
    """
    Run GSVA (via ssgsea, as it supports sparse matrices) on the prepared dataset.
    ssgsea requires log(CPM) (see here: https://github.com/rcastelo/GSVA/issues/59)

    https://gseapy.readthedocs.io/en/latest/index.html
    https://bioconductor.org/packages/devel/bioc/vignettes/GSVA/inst/doc/GSVA.html#6_Example_applications
    """
    input:
        sparse_matrix=rules.prepare_gsva_dataset.output.sparse_matrix,
        colnames=rules.prepare_gsva_dataset.output.colnames,
        rownames=rules.prepare_gsva_dataset.output.rownames,
        geneset=rules.download_genesets.output.geneset_gmt
    output:
        gsva_csv=PROJECT_DIR / "results" / "{dataset}" / "tmp" / "gsva_{library}.csv"  # make temporary
    threads: 122
    resources:
        mem_mb=600000,
        slurm="cpus-per-task=122"  # 124 is max
    conda:
        "../envs/gsva.yaml"
    notebook: "../notebooks/gsva.r.ipynb"

rule combine_gsvas:
    input:
        expand(rules.gsva.output.gsva_csv, library=config["genesets"], dataset=["{dataset}"]),
    output:
        combined_gsvas=PROJECT_DIR / config["paths"]["gsva"]["result"]
    conda: "cellwhisperer"
    script: "../scripts/combine_gsvas.py"
