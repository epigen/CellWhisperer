BOWEL_DISEASE_PLOTS = PROJECT_DIR / "results" / "plots" / "bowel_disease"

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
        plot=BOWEL_DISEASE_PLOTS / "stem_cell_contribution{target_cluster,[^/]*}.svg"
    conda:
        "cellwhisperer"
    resources:
        mem_mb=50000,
        slurm=slurm_gres()
    params:
        search_term="stem cells",
    notebook:
        "../notebooks/bowel_disease_stem_cell_contribution.py.ipynb"

rule bowel_disease_conventional_analysis:
    """
    NOTE: This notebook was executed interactively and does not run out of the box within snakemake. We provide it here as "pseudocode" to be run interactively with snakemake's `--edit-notebook` option.
    """
    input:
        matrix_tsv=HTTP.remote("https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE116222&format=file&file=GSE116222%5FExpression%5Fmatrix%2Etxt%2Egz", keep_local=True)[0],
        cellwhisperer_dataset=HTTP.remote(f"{config['precomputing_base_url']}/datasets/bowel_disease/cellxgene.h5ad", keep_local=True)[0],
    output:
        umap_raw=BOWEL_DISEASE_PLOTS / "umap_raw.svg",
        umap_raw_legend=BOWEL_DISEASE_PLOTS / "umap_raw_legend.svg",
        umap_integrated=BOWEL_DISEASE_PLOTS / "umap_integrated.svg",
        umap_integrated_legend=BOWEL_DISEASE_PLOTS / "umap_integrated_legend.svg",
        umap_annotated=BOWEL_DISEASE_PLOTS / "umap_annotated.svg",
        umap_annotated_legend=BOWEL_DISEASE_PLOTS / "umap_annotated_legend.svg",
        umap_cellwhisperer=BOWEL_DISEASE_PLOTS / "umap_cellwhisperer.svg",
        umap_cellwhisperer_legend=BOWEL_DISEASE_PLOTS / "umap_cellwhisperer_legend.svg",
        stemness_score_plot=BOWEL_DISEASE_PLOTS / "stemness_score_plot.svg",
        rank_plot=BOWEL_DISEASE_PLOTS / "rank_plot.svg",
    params:
        celltypist_model="Cells_Intestinal_Tract.pkl",  # downloaded automaticallyy
        stemness_markers='NES,KDM5B,ZFP42,CTNNB1,ZSCAN4,EPAS1,EZH2,HIF1A,MYC,NOTCH1,POU5F1,BMI1,SOX2,TWIST1,NANOG,LGR5,PROM1,KLF4,ABCG2,CD34,CD44'.split(','),  # Malta et al. 2018
        condition_palette={  # derived from https://cellwhisperer.bocklab.org/colonic_epithelium/
            'healthy': '#6e40aa',
            'inflamed': '#ff8c38',
            'non-inflamed': '#28ea8d'
        }
    conda:
        "../../shared/envs/sctools.yaml"
    resources:
        mem_mb=100000,
        slurm=slurm_gres()
    threads: 8
    log:
        notebook="../logs/bowel_disease_conventional_analysis.ipynb",
    notebook:
        "../notebooks/bowel_disease_conventional_analysis.py.ipynb"

rule fig5_all:
    input:
        expand(rules.bowel_disease_stem_cell_contribution.output.plot, target_cluster=["", "Cycling ileal epithelial precursor cells"]),
        # rules.bowel_disease_conventional_analysis.output.umap_raw,
