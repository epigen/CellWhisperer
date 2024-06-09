rule compute_gene_normalizers:
    """
    Compute the gene normalizers (in log scale) for each gene across all samples.

    `np.log(gene + 1).mean()`

    Requires a lot of RAM to be able to transpose the sparse matrix (required for efficient computation)
    """
    input:
        read_count_table=PROJECT_DIR / config["paths"]["read_count_table"],
    output:
        gene_mean_log1ps=PROJECT_DIR / "results" / "gene_normalizers" / "{dataset}.pickle"
    resources:
        mem_mb=500000,
        slurm="cpus-per-task=64"
    threads: 64
    conda:
        "cellwhisperer"
    script:
        "../scripts/compute_gene_normalizers.py"

rule process_full_dataset:
    """
    Run CellWhisperer on the read_count_table (provided) dataset and store the outputs in an npz file.
    """
    input:
        read_count_table=PROJECT_DIR / config["paths"]["read_count_table"],
        model=ancient(PROJECT_DIR / config["paths"]["jointemb_models"] / "{model}.ckpt"),
        geneformer_model=ancient(PROJECT_DIR / "resources" / "geneformer-12L-30M" / "pytorch_model.bin")  # making sure that geneformer is downloaded
    output:
        model_outputs=protected(PROJECT_DIR / config["paths"]["model_processed_dataset"]),
    resources:
        mem_mb=600000,  # could be made more efficient...
        slurm=f"cpus-per-task=5 gres=gpu:{GPU_TYPE}:1 qos={GPU_TYPE} partition=gpu"
    threads: 64
    log:
        notebook="../logs/notebooks/process_full_dataset_{dataset}_{model}.py.ipynb",
        log_file="../logs/process_full_dataset_{dataset}_{model}.log"
    conda:
        "cellwhisperer"
    notebook:
        "../notebooks/process_full_dataset.py.ipynb"

rule combine_processed_data:
    """
    Combine the processed data from all datasets used in LLaVA (CellWhisperer LLM) to be able to train and validate on them.

    Since we use `orig_ids` to match the data, we can't simply concatenate the arrays.
    """
    input:
        expand(rules.process_full_dataset.output.model_outputs, dataset=["archs4_geo", "tabula_sapiens", "cellxgene_census", "human_disease"], model="{model}"),
    output:
        combined=PROJECT_DIR / config["paths"]["llava"]["combined_processed_data"]
    resources:
        mem_mb=100000
    run:
        import numpy as np
        datas = [dict(np.load(dataset_fn, allow_pickle=True)) for dataset_fn in input]

        assert len(set.union(*[set(d["orig_ids"]) for d in datas])) == sum(len(d["orig_ids"]) for d in datas), "No duplicate ids allowed"

        combined = {k: np.concatenate([d[k] for d in datas]) for k in datas[0].keys()}
        np.savez(output.combined, **combined)


rule compute_top_genes:
    """
    Compute the top genes for each sample based on the gene normalizers.

    All genes are considered such that also genes may come up that are not reflected Geneformer's vocabulary. This may be fine, since these non-represented genes are likely impacting other, represented, genes.

    Requires a lot of RAM to be able to transpose the sparse matrix (required for efficient computation)

    Takes about ~1 hour for archs4_geo
    """

    input:
        read_count_table=PROJECT_DIR / config["paths"]["read_count_table"],
        gene_normalizers=rules.compute_gene_normalizers.output.gene_mean_log1ps,
        # HTTP.remote("https://huggingface.co/ctheodoris/Geneformer/resolve/main/geneformer/gene_median_dictionary.pkl", keep_local=True)[0],
    output:
        top_genes=PROJECT_DIR / "results" / "{dataset}" / "top_genes.parquet"
    params:
        top_n_genes=100,
    resources:
        mem_mb=500000,
        slurm="cpus-per-task=2"
    conda:
        "cellwhisperer"
    notebook:
        "../notebooks/compute_top_genes.py.ipynb"
