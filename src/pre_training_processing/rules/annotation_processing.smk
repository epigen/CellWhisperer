from collections import defaultdict

scattergather:
    split=256

rule prepare_requests:
    """
    Open the structured annotations dataframe and extract the sample with the appropriate ID. For performance reasons, extract all requested at once.

    NOTE: if we wanted to introduce variability in the requests, we could do it here in function of the replicate number.
    """
    input:
        # structured_annotations = lambda wildcards: ancient(PROJECT_DIR / config["paths"]["structured_annotations"].format(dataset=wildcards.dataset)),
        structured_annotations = ancient(PROJECT_DIR / config["paths"]["structured_annotations"])
    output:
        yaml_splits=scatter.split(PROJECT_DIR / "results/pre_training_processing/requests/{{dataset,[^/]+}}/{{replicate}}/{scatteritem}.yaml")
        # optionally add json
    run:
        import json
        import yaml

        with open(input.structured_annotations) as f:
            annotations = json.load(f)

        # wildcards.scatteritem contains i-of-n.yaml
        for split_fn in output.yaml_splits:
            split_i, split_n = map(int, Path(split_fn).stem.split('-of-'))
            split_i -= 1  # 0-indexing
            # take the i-th split from annotations:
            split_annotations = {k: v for i, (k, v) in enumerate(annotations.items()) if i % split_n == split_i}

            # write the split to a file
            with open(split_fn, "w") as f:
                yaml.dump(split_annotations, f)

rule process_annotation_local:
    """
    """
    input:
        model=PROJECT_DIR / config["model_name_path_map"]["mixtral"],
        instruction="prompts/process_annotations_few_shot.txt",
        yaml_split=PROJECT_DIR / "results/pre_training_processing/requests/{dataset,[^/]+}/{replicate}/{scatteritem}.yaml",
        few_shot_prompts=expand("prompts/few_shot_samples/{i}_request.json", i=range(9)),
        few_shot_responses=expand("prompts/few_shot_samples/{i}_response.json", i=range(9)),
        json_schema="prompts/output_schema.json"
    output:
        processed_annotation = protected(PROJECT_DIR / "results" / "pre_training_processing" / "processed" / "{dataset}" / "{replicate}" / "{scatteritem}.csv"),  # I marked this as protected as it might be costly to produce
        requests = (PROJECT_DIR / "results" / "pre_training_processing" / "formatted_requests" / "{dataset}" / "{replicate}" / "{scatteritem}.yaml")
    params:
        study_specific_fields="treatment_protocol series_summary series_design growth_protocol sample_type mapped_ontology_terms study_description study_title".split(" ")
    resources:
        mem_mb=100000,
        slurm=slurm_gres(num_cpus=20)
        # slurm=slurm_gres(num_cpus=20)
    conda: "llama_cpp"
    script: "../scripts/process_annotations_local.py"

rule aggregate_processed:
    """
    Read in all the generated annotations and aggregate them into a single JSON file
    """
    input:
        yaml_splits=[split.format(dataset="{dataset}", replicate=replicate)
                     for split in gather.split(PROJECT_DIR / "results" / "pre_training_processing" / "processed" / "{{dataset}}" / "{{replicate}}" / "{scatteritem}.csv")
                     for replicate in REPLICATES],
    output:
        single=PROJECT_DIR / config["paths"]["processed_annotations"],
        multi=PROJECT_DIR / config["paths"]["processed_multi_annotations"],
    script:
        "../scripts/aggregate_processed.py"
