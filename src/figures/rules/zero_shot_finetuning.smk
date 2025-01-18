from pathlib import Path
import subprocess


PROJECT_DIR = Path(subprocess.check_output("git rev-parse --show-toplevel", shell=True).decode("utf-8").strip())
FINETUNE_RESULTS_DIR = PROJECT_DIR / "results" / "finetuning_eval"
configfile: PROJECT_DIR / "config.yaml"
EVAL_DATASETS = [d for d, cols in config["metadata_cols_per_zero_shot_validation_dataset"].items() if "celltype" in cols]
MODELS = ["geneformer"]  # "scgpt", "uce"
GPU_TYPE = "a100"

rule finetune_scfm:
    """
    Fine-tune on cellxgene_census (due to their consistent annotations)
    """
    input:
        model_weights=lambda wildcards: PROJECT_DIR / config["model_name_path_map"][wildcards.model],
        training_data=PROJECT_DIR / config["paths"]["read_count_table"].format(dataset="cellxgene_census"),
    output:
        model_weights=FINETUNE_RESULTS_DIR / "{model}" / "finetuned.pt"
    params:
        label_col="celltype",  # We only have this one consistently
        use_replicates=False,
        use_aggregated=True,
        num_epochs=2,  # TODO 16?
        batch_size=64,
        freeze_fm=True,
    resources:
        mem_mb=350000,
        slurm=f"cpus-per-task=5 gres=gpu:{GPU_TYPE}:1 qos={GPU_TYPE} partition=gpu"
    conda:
        lambda wildcards: "cellwhisperer" if wildcards.model in ["geneformer", "scgpt"] else "../../envs/uce.yaml"
    log:
        notebook="logs/finetune_scfm_{model}.ipyb"
    notebook:
        "../notebooks/finetune_scfm_{wildcards.model}.py.ipynb"

rule transfer_labels:
    """
    Transfer labels from training data to evaluation data
    """
    input:
        training_data=PROJECT_DIR / config["paths"]["read_count_table"].format(dataset="cellxgene_census"),
        eval_data=PROJECT_DIR / config["paths"]["read_count_table"],
    output:
        transfered_labels=FINETUNE_RESULTS_DIR / "{dataset}" / "transfered_labels.csv",
    params:
        label_col="celltype",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-2024-11-20"
    conda:
        "cellwhisperer"
    log:
        notebook="logs/transfer_labels_{dataset}.ipynb"
    notebook:
        "../notebooks/transfer_labels.py.ipynb"

rule evaluate_scfm:
    """
    Evaluate fine-tuned model on different datasets
    """
    input:
        model=rules.finetune_scfm.output.model_weights,
        eval_data=PROJECT_DIR / config["paths"]["read_count_table"],
        transfered_labels=rules.transfer_labels.output.transfered_labels,
    output:
        predictions_raw=FINETUNE_RESULTS_DIR / "{model}" / "{dataset}" / "predictions_raw.csv",  # this is the same for all datasets
        predictions=FINETUNE_RESULTS_DIR / "{model}" / "{dataset}" / "predictions.csv",
        performance=FINETUNE_RESULTS_DIR / "{model}" / "{dataset}" / "performance.csv",
    params:
        batch_size=128,
        label_col="celltype",
    conda:
        "cellwhisperer"
    log:
        notebook="logs/evaluate_scfm_{model}_{dataset}.ipynb"
    notebook:
        "../notebooks/evaluate_scfm.py.ipynb"


rule all:
    input:
        expand(rules.evaluate_scfm.output[0], model=MODELS, dataset=EVAL_DATASETS, label_col=["celltype"])  # TODO use all properties

    default_target: True
