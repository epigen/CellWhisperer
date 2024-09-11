from snakemake.remote.HTTP import RemoteProvider as HTTPRemoteProvider
HTTP = HTTPRemoteProvider()

rule bowel_disease_stem_cell_contribution:
    """
    leads to p < 1e-37
    """
    input:
        # dataset=PROJECT_DIR / "results" / "bowel_disease" / CLIP_MODEL / "cellxgene.h5ad",  # Our pipeline could process the file from scratch, but due to stochasticity, there will be slight mismatches with the plots in the manuscript
        dataset=HTTP.remote(f"{config['precomputing_base_url']}/datasets/bowel_disease/cellxgene.h5ad", keep_local=True)[0],
        model=PROJECT_DIR / config["paths"]["jointemb_models"] / f"{CLIP_MODEL}.ckpt",  # needed for the keywords
        mpl_style=ancient(PROJECT_DIR / config["plot_style"])
    output:
        plot=PROJECT_DIR / "results" / "plots" / "bowel_disease" / "stem_cell_contribution{target_cluster,[^/]*}.svg"
    conda:
        "cellwhisperer"
    resources:
        mem_mb=50000,
        slurm="cpus-per-task=5 gres=gpu:a100:1 qos=a100 partition=gpu"
    params:
        search_term="stem cells",
    notebook:
        "../notebooks/bowel_disease_stem_cell_contribution.py.ipynb"
