"""
Details: https://github.com/epigen/cellwhisperer/issues/385
"""
import subprocess
from pathlib import Path
PROJECT_DIR = Path(subprocess.check_output("git rev-parse --show-toplevel", shell=True).decode("utf-8").strip())

# configfile: PROJECT_DIR / "config.yaml"

rule cw_transcriptome_term_scores:
    """
    - Compute the term-based match scores using (CW)
        - Embed all terms (present in gsva_results)
        - Use processed dataset and get cos_sim between each cell and each term
    """
    input:
        processed_dataset=rules.process_full_dataset.output.model_outputs,  # PROJECT_DIR / config["paths"]["model_processed_dataset"],
        gsva_results=PROJECT_DIR / config["paths"]["gsva"]["result"],
        model=PROJECT_DIR / config["paths"]["jointemb_models"] / "{model}.ckpt",  # needed for the keywords
    output:
        cw_transcriptome_term_scores=PROJECT_DIR / config["paths"]["gsva"]["cw_transcriptome_term_scores"],
    conda:
        "cellwhisperer"
    resources:
        mem_mb=40000,
        slurm="cpus-per-task=5 gres=gpu:a100:1 qos=a100 partition=gpu"
    log:
        notebook="../logs/gsva_correlation_{model}_{dataset}.log"
    notebook:
        "../notebooks/cw_transcriptome_term_scores.py.ipynb"

rule plot_gsva_correlations:
    """
    - Plot the correlation between the top term and the cells
    - Plot the correlation between the libraries
    - Correlate (for each (pseudo)cell, sample)
    """
    input:
        cw_transcriptome_term_scores=rules.cw_transcriptome_term_scores.output.cw_transcriptome_term_scores,
        gsva_results=PROJECT_DIR / config["paths"]["gsva"]["result"],
        mpl_style=ancient(PROJECT_DIR / config["plot_style"])
    output:
        gsva_correlation_results=PROJECT_DIR / config["paths"]["gsva"]["correlation"],
        top_term_correlation=PROJECT_DIR / config["paths"]["gsva"]["plots"] / "gsva_top_term_correlation.svg",
        library_correlations=PROJECT_DIR / config["paths"]["gsva"]["plots"] / "gsva_library_correlations.svg",
        term_level_correlations=PROJECT_DIR / config["paths"]["gsva"]["plots"] / "gsva_term_level_correlations.svg",
        omim_correlations=PROJECT_DIR / config["paths"]["gsva"]["plots"] / "gsva_omim_correlations.svg",
        cw_binarized_gsva_scores=PROJECT_DIR / config["paths"]["gsva"]["plots"] / "cw_binarized_gsva_scores.svg",
        library_ks_statistics=PROJECT_DIR / config["paths"]["gsva"]["plots"] / "gsva_library_ks_statistics.svg",  # also binarized
        cherry_picked_examples=PROJECT_DIR / config["paths"]["gsva"]["plots"] / "gsva_cherry_picked_examples.svg",
    params:
        selected_top_term="Pluripotent Stem Cells"  # not the one we want..
    conda:
        "cellwhisperer"
    resources:
        mem_mb=200000,
        slurm="cpus-per-task=2"
    log:
        notebook="../logs/plot_gsva_correlation_{dataset}_{model}.ipynb"
    notebook:
        "../notebooks/plot_gsva_correlations.py.ipynb"
