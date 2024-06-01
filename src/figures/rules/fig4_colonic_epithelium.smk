rule bowel_disease_stem_cell_contribution:
    input:
        dataset=PROJECT_DIR / "results" / "bowel_disease" / CLIP_MODEL / "cellxgene.h5ad",
        model=PROJECT_DIR / config["paths"]["jointemb_models"] / f"{CLIP_MODEL}.ckpt",  # needed for the keywords
        mpl_style=ancient(PROJECT_DIR / config["plot_style"])
    output:
        plot=PROJECT_DIR / "results" / "plots" / "bowel_disease" / "stem_cell_contribution.svg"
    conda:
        "cellwhisperer"
    resources:
        mem_mb=50000,
        slurm="cpus-per-task=5 gres=gpu:a100:1 qos=a100 partition=gpu"
    params:
        search_term="stem cell",
        target_cluster="Cycling ileal epithelial precursor cells"
    notebook:
        "../notebooks/bowel_disease_stem_cell_contribution.py.ipynb"
