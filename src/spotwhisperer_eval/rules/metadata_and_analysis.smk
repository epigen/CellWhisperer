# Metadata extraction and per-class analyses
from pathlib import Path
import glob


def samples_for_dataset(dataset_name):
    """
    Resolve sample indices for datasets using the full_dataset_multi pattern.
    Returns sorted list of sample indices.
    """
    pattern_path = PROJECT_DIR / config["paths"]["full_dataset_multi"].format(dataset=dataset_name, i="*")
    matching_files = glob.glob(str(pattern_path))

    sample_indices = []
    for file_path in matching_files:
        filename = Path(file_path).stem
        if filename.startswith("full_data_"):
            i_value = filename[len("full_data_"):]
            sample_indices.append(i_value)

    if len(sample_indices) == 0:
        raise ValueError(f"No file found for {dataset_name}")
    return sorted(sample_indices)


rule extract_sample_metadata:
    """
    Export per-sample metadata from retrieval datasets into report files.
    """
    input:
        dataset_files=lambda wildcards: [
            PROJECT_DIR / config["paths"]["full_dataset_multi"].format(dataset=component, i=i)
            for component in wildcards.dataset.split("__")
            if component in ["hest1k", "quilt1m"]
            for i in samples_for_dataset(component)
        ] + [
            PROJECT_DIR / config["paths"]["full_dataset"].format(dataset=component)
            for component in wildcards.dataset.split("__")
            if component not in ["hest1k", "quilt1m"]
        ]
    output:
        metadata=report(BENCHMARKS_DIR / "sample_metadata" / "{dataset}" / "sample_metadata.csv", category="per_class_analysis", subcategory=lambda wildcards: DATASET_PAIR_MAPPING[wildcards.dataset], labels={"Analysis": "Sample-level metadata", "Format": "csv"}),
    params:
        project_dir=PROJECT_DIR,
    conda:
        "cellwhisperer"
    resources:
        mem_mb=50000,
        slurm="cpus-per-task=4"
    script:
        "../scripts/extract_sample_metadata.py"


rule retrieval_per_class_analysis:
    """
    Collect per-class retrieval scores for the bimodal_bridge model into report and plot.
    """
    input:
        model_results=lambda wildcards: rules.spotwhisperer_test.output.individual_clip_scores.format(
            dataset_combo=MODEL_MAPPINGS[wildcards.dataset]["bimodal_bridge"],
            test_dataset=wildcards.dataset,
        )
    output:
        analysis=report(BENCHMARKS_DIR / "retrieval_per_class" / "{dataset}" / "per_class_analysis.csv", category="per_class_analysis", subcategory=lambda wildcards: DATASET_PAIR_MAPPING[wildcards.dataset], labels={"Analysis": "Retrieval per class", "Format": "csv"}),
        plot=report(BENCHMARKS_DIR / "retrieval_per_class" / "{dataset}" / "per_class_analysis.pdf", category="per_class_analysis", subcategory=lambda wildcards: DATASET_PAIR_MAPPING[wildcards.dataset], labels={"Analysis": "Retrieval per class", "Format": "plot"}),
    params:
        model_mappings=lambda wildcards: MODEL_MAPPINGS[wildcards.dataset],
    conda:
        "cellwhisperer"
    resources:
        mem_mb=20000,
        slurm="cpus-per-task=2"
    log:
        notebook="../logs/retrieval_per_class_analysis_{dataset}.ipynb"
    notebook:
        "../notebooks/retrieval_per_class_analysis.py.ipynb"
