

MARKER_BASED_RESULTS = PROJECT_DIR / "results" / "marker_based_celltypes"
MARKER_BASED_PLOTS = PROJECT_DIR / "results" / "plots" / "marker_based_celltypes"

rule curate_marker_database_download:
    input:
        HTTP.remote(f"{config['precomputing_base_url']}/datasets/{{dataset}}/marker_based_celltypes/prepared_markers.csv", keep_local=False)[0],
    output:
        prepared_markers=MARKER_BASED_RESULTS / "{dataset}" / "prepared_markers.csv",
    run:
        import shutil
        shutil.copy(input[0], output.prepared_markers)

# rule curate_marker_database:
#     """
#     Download and curate CellMarker 2.0 database https://academic.oup.com/nar/article/51/D1/D870/6775381

#     NOTE: For now, we want a match for each of the evaluation cell types.

#     """
#     input:
#         cellmarker2_human=HTTP.remote("http://117.50.127.228/CellMarker/CellMarker_download_files/file/Cell_marker_Human.xlsx", keep_local=True)[0],
#         eval_data=PROJECT_DIR / config["paths"]["read_count_table"],
#     output:
#         prepared_markers=MARKER_BASED_RESULTS / "{dataset}" / "prepared_markers.csv",
#     conda:
#         "cellwhisperer"
#     # params:
#         # organ_selector=lambda wildcards:  TODO implement in notebook
#     resources:
#         mem_mb=200000,
#         slurm="cpus-per-task=2"
#     params:
#         openai_api_key=os.getenv("OPENAI_API_KEY"),
#         model="gpt-4o-2024-11-20"
#     log:
#         notebook="../logs/curate_marker_database_{dataset}.ipynb",
#         log="../logs/curate_marker_database_{dataset}.log"
#     notebook:
#         "../notebooks/curate_marker_database.py.ipynb"


rule cell_assign:
    """
    """
    input:
        read_count_table=PROJECT_DIR / config["paths"]["read_count_table"],
        prepared_markers=rules.curate_marker_database_download.output.prepared_markers,
    output:
        predictions_raw=MARKER_BASED_RESULTS / "{dataset}" / "predictions_raw.csv",
        # predictions=MARKER_BASED_RESULTS / "{dataset}" / "predictions.csv",
        performance=MARKER_BASED_RESULTS / "{dataset}" / "performance.csv",
    params:
        seed=42,
        label_col="celltype",
    conda:
        "cellwhisperer"
    resources:
        mem_mb=500000,
        slurm=slurm_gres()
    log:
        notebook="../logs/cell_assign_{dataset}.ipynb",
    notebook:
        "../notebooks/cell_assign.py.ipynb"
