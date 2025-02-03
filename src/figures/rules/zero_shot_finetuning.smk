from pathlib import Path
import subprocess


PROJECT_DIR = Path(subprocess.check_output("git rev-parse --show-toplevel", shell=True).decode("utf-8").strip())
FINETUNE_RESULTS_DIR = PROJECT_DIR / "results" / "finetuning_eval"
GPU_TYPE = "a100-sxm4-80gb"  # we actually need 80gb for `unfrozen` (at least for UCE) and more
TRAINING_OPTIONS=["frozen", "frozen_singlecells", "unfrozen"]

rule finetune_scfm:
    """
    Fine-tune on cellxgene_census (due to their consistent annotations)
    """
    input:
        model_weights=lambda wildcards: PROJECT_DIR / config["model_name_path_map"][wildcards.model],
        training_data=PROJECT_DIR / config["paths"]["read_count_table"].format(dataset="cellxgene_census"),
    output:
        model_weights=FINETUNE_RESULTS_DIR / "{model}" / "finetuned_{training_options}.pt"
    params:
        label_col="cell_type",  # We only have this one consistently (and it's with '_' in this dataset)
        use_replicates=lambda wildcards: "singlecells" in wildcards.training_options,
        num_epochs=8,
        batch_size=16,  # NOTE: the frozen ones were trained with 64 (translating to a lower learning rate)
        learning_rate=lambda wildcards: 1e-4 if "unfrozen" not in wildcards.training_options else 1e-5,  # NOTE: 1e-5 was only introduced for uce. The others did not yet 'benefit' from it
        freeze_fm=lambda wildcards: "unfrozen" not in wildcards.training_options,
    resources:
        mem_mb=lambda wildcards: 800000 if wildcards.model == "uce" else 350000,
        slurm=f"cpus-per-task=5 gres=gpu:a100-sxm4-80gb:1 qos=a100-sxm4-80gb partition=gpu"
    conda:
        "cellwhisperer"
        # lambda wildcards: "cellwhisperer" if wildcards.model in ["geneformer", "scgpt"] else "../../envs/uce.yaml"
    log:
        notebook="logs/finetune_scfm_{model}_{training_options}.ipynb"
    notebook:
        "../notebooks/finetune_scfm.py.ipynb"

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
    resources:
        mem_mb=350000,
        slurm="cpus-per-task=2"
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
        predictions_raw=FINETUNE_RESULTS_DIR / "{model}" / "{dataset}" / "predictions_raw_{training_options}.csv",  # this is the same for all datasets
        predictions=FINETUNE_RESULTS_DIR / "{model}" / "{dataset}" / "predictions_{training_options}.csv",
        performance=FINETUNE_RESULTS_DIR / "{model}" / "{dataset}" / "performance_{training_options}.csv",
    params:
        batch_size=64,
        label_col="celltype",
    resources:
        mem_mb=lambda wildcards: 450000 if wildcards.model == "uce" else 300000,
        slurm=f"cpus-per-task=5 gres=gpu:a100:1 qos=a100 partition=gpu"
    conda:
        "cellwhisperer"
    log:
        notebook="logs/evaluate_scfm_{model}_{dataset}_{training_options}.ipynb",
        progress="logs/evaluate_scfm_{model}_{dataset}_{training_options}.log"
    notebook:
        "../notebooks/evaluate_scfm.py.ipynb"
