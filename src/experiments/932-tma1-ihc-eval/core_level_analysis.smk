"""
Snakemake rules for core-level analysis: correlating base model performance
(retrieval, contrastive loss) with decoder performance per TMA core.

Include from main Snakefile:
    include: "core_level_analysis.smk"

Expects variables from the main Snakefile:
    PROJECT_DIR, RESULTS_DIR, SCRIPTS_DIR, EXPERIMENT_DIR,
    CELLWHISPERER_CHECKPOINT, EVAL_TMAS
"""

CORE_LEVEL_DIR = RESULTS_DIR / "metrics" / "core_level"
CORE_PLOTS_DIR = RESULTS_DIR / "plots" / "core_level_correlation"


# Rule: all core-level analysis targets
rule core_level_all:
    input:
        combined = CORE_LEVEL_DIR / "core_metrics_combined.csv",
        corr_matrix = CORE_LEVEL_DIR / "correlation_matrix.csv",
        plots = CORE_PLOTS_DIR,


# Rule: compute per-core retrieval and contrastive loss using base model
rule compute_core_base_metrics:
    input:
        checkpoint = CELLWHISPERER_CHECKPOINT,
        predictions = RESULTS_DIR / "predictions" / "{tma}_predictions.h5ad",
    output:
        retrieval = CORE_LEVEL_DIR / "{tma}_core_retrieval.csv",
        loss = CORE_LEVEL_DIR / "{tma}_core_loss.csv",
    params:
        dataset_name = lambda wildcards: f"lymphoma_cosmx_large_{wildcards.tma}",
        script = SCRIPTS_DIR / "compute_core_base_metrics.py",
    conda:
        "cellwhisperer"
    threads: 10
    resources:
        mem_mb = 80000,
        slurm = slurm_gres("medium", num_cpus=10, time="02:00:00")
    shell: """
        python {params.script} \
            --checkpoint {input.checkpoint} \
            --dataset_name {params.dataset_name} \
            --predictions {input.predictions} \
            --batch_size 64 \
            --nproc 8 \
            --min_cells 50 \
            --output_retrieval {output.retrieval} \
            --output_loss {output.loss}
    """


# Rule: compute per-core decoder performance from predictions
rule compute_core_decoder_performance:
    input:
        predictions = RESULTS_DIR / "predictions" / "{tma}_predictions.h5ad",
    output:
        metrics = CORE_LEVEL_DIR / "{tma}_core_decoder.csv",
    params:
        script = SCRIPTS_DIR / "compute_core_decoder_performance.py",
    conda:
        "cellwhisperer"
    threads: 4
    resources:
        mem_mb = 40000,
    shell: """
        python {params.script} \
            --predictions {input.predictions} \
            --top_n 500 \
            --min_cells 50 \
            --output {output.metrics}
    """


# Rule: correlate base model metrics with decoder performance
rule correlate_core_metrics:
    input:
        retrieval = expand(CORE_LEVEL_DIR / "{tma}_core_retrieval.csv", tma=EVAL_TMAS),
        loss = expand(CORE_LEVEL_DIR / "{tma}_core_loss.csv", tma=EVAL_TMAS),
        decoder = expand(CORE_LEVEL_DIR / "{tma}_core_decoder.csv", tma=EVAL_TMAS),
    output:
        combined = CORE_LEVEL_DIR / "core_metrics_combined.csv",
        corr_matrix = CORE_LEVEL_DIR / "correlation_matrix.csv",
        plots = directory(CORE_PLOTS_DIR),
    params:
        script = SCRIPTS_DIR / "correlate_core_metrics.py",
    conda:
        "cellwhisperer"
    threads: 2
    resources:
        mem_mb = 16000,
    shell: """
        python {params.script} \
            --retrieval_csvs {input.retrieval} \
            --loss_csvs {input.loss} \
            --decoder_csvs {input.decoder} \
            --output_combined {output.combined} \
            --output_corr_matrix {output.corr_matrix} \
            --output_plots {output.plots}
    """
