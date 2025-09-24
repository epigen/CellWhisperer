# remove later:
include: "../../shared/config.smk"


DEVELOPMENT_PLOTS = PROJECT_DIR / "results" / "plots" / "development_analysis"

rule marker_genes_db:
    input:
        HTTP.remote(config["precomputing_base_url"] + "/datasets/development/aba7721_tabless1-s16.xlsx", keep_local=False)[0]  # "https://www.science.org/doi/suppl/10.1126/science.aba7721/suppl_file/aba7721_tabless1-s16.xlsx"
    output:
        PROJECT_DIR / "resources/development/aba7721_tabless1-s16.xlsx"
    run:
        import shutil
        shutil.copy(input[0], output[0])

rule development_analysis:
    """
    Performs post-processing analysis on AnnData objects previously processed with CellWhisperer,
    focusing on developmental time courses, organ emergence, marker gene identification, and comparison
    with reference datasets and literature.
    """
    input:
        processed_adata=PROJECT_DIR / f"results/development/{CLIP_MODEL}/cellxgene.h5ad",
        marker_genes=rules.marker_genes_db.output[0],
        model=PROJECT_DIR / config["paths"]["jointemb_models"] / f"{CLIP_MODEL}.ckpt",
        mpl_style=ancient(PROJECT_DIR / config["plot_style"])
    output:
        result_dir=directory(DEVELOPMENT_PLOTS / "output")
    conda:
        "cellwhisperer"
    resources:
        mem_mb=150000,
        slurm=slurm_gres()
    log:
        notebook="../logs/development_analysis.ipynb"
    notebook:
        "../notebooks/development/development_analysis.ipynb"

rule fig3_main:
    input:
        rules.development_analysis.output.result_dir
