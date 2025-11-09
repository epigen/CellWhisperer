RESULTS_DIR = PROJECT_DIR / "results" / "pre_training_processing"

rule transcriptome_representation:
    """
    Simply normalize the data and bring it into the expected npz format

    """
    input:
        read_count_table=PROJECT_DIR / config["paths"]["read_count_table"],
    output:
        representation=RESULTS_DIR / "{dataset}" / "transcriptome_representation.npz"
    conda:
        "../envs/pydata.yaml"
    params:
        n_dimensions=768  # match the annotation dimensionality
    resources:
        mem_mb=800000,
    script:
        "../scripts/transcriptome_representation.py"


rule annotation_representation:
    """
    Use BioBERT to calculate representation for the text data

    NOTE: It would have been better to take the second replicate

    """
    input:
        annotations=PROJECT_DIR / config["paths"]["processed_annotations"],
    output:
        representation=RESULTS_DIR / "{dataset}" / "annotation_representation.npz"
    conda:
        "cellwhisperer"
    resources:
        mem_mb=400000,
        slurm=slurm_gres()
    script:
        "../scripts/annotation_representation.py"


rule local_density_to_sample_weight:
    """
    Calculate the local density of the representation and store as sample weight
    https://www.nature.com/articles/s41587-020-00801-7

    NOTE: we could also test how the *embedding*-density performs in comparison
    """
    input:
        representation=RESULTS_DIR / "{dataset}" / "{modality}_representation.npz"
    output:
        weight=protected(RESULTS_DIR / "{dataset}" / "{modality}_weights.npz"),
        plot_orig_radii=RESULTS_DIR / "{dataset}" / "{modality}_weights_dist.png",
        orig_radii=RESULTS_DIR / "{dataset}" / "{modality}_orig_radii.npz",
    conda:
        "../envs/pydata.yaml"
    log:
        "logs/local_density_to_sample_weight_{dataset}_{modality}.log"
    resources:
        mem_mb=40000,
        slurm="cpus-per-task=100"
    script:
        "../scripts/local_density_to_sample_weight.py"
