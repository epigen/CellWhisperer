# MUSK evaluation pipeline
# Based on the structure of spotwhisperer.smk but adapted for MUSK benchmarks
# Note: This pipeline uses the MUSK benchmark framework which requires vision-language models
# The original SpotWhisperer model may need adaptation to work with pathology image tasks

MUSK_RESULTS = PROJECT_DIR / "results/musk_evaluation"
MUSK_MODEL_RESULTS = MUSK_RESULTS / "{model}"

# MUSK evaluation datasets and tasks
# MUSK_DATASETS = ["pathmmu_retrieval", "unitopatho_retrieval", "skin", "pannuke", "unitopatho"]
# MUSK_TASKS = ["zeroshot_retrieval", "zeroshot_classification", "image_retrieval", "linear_probe"]

# Configuration for dataset root - can be overridden in config.yaml or via command line
DATASET_ROOT = PROJECT_DIR / config["paths"]["musk"]["datasets"]  # points to scratch, because it is many files

STATIC_MODELS_TXT = "modules/MUSK/benchmarks/models.txt"

rule download_musk_datasets:
    """
    Download/populate MUSK datasets using Snakemake remote provider.
    Produces a marker file to indicate the dataset directory is ready.

    NOTE: there is either the precomputed dataset, or there is the list of all large-scale datasets
    """
    output:
        directory=directory(DATASET_ROOT),
        tmp_file=temporary(DATASET_ROOT / "musk_datasets.zip")
    params:
        file_id="1FCGRn6mtdrw8l3WAR_U76V0eRBnsQxD1"
    conda:
        "../../../envs/gdown.yaml"
    shell: """
        gdown --id {params.file_id} -O {output.tmp_file}
        unzip -o {output.tmp_file} -d {output.directory}
        mv {output.directory}/downstreams_demo/* {output.directory}/
        rmdir {output.directory}/downstreams_demo/
    """

def model_descriptor_fn(wildcards, input):
    return "spotwhisperer_local,{}".format(input.model)


rule musk_zero_shot_classification:
    """
    Zero-shot classification evaluation using MUSK benchmark
    """
    input:
        model=PROJECT_DIR / config["paths"]["jointemb_models"] / "{model}.ckpt",
        dataset_dir=ancient(DATASET_ROOT)
    output:
        results=MUSK_MODEL_RESULTS / "results" / "zeroshot_classification_{dataset}_seed{seed}.json"
    threads: 8
    params:
        batch_size=256,
        model_descriptor=model_descriptor_fn
    wildcard_constraints:
        dataset="skin|pannuke|unitopatho",
        seed="\\d+"
    conda:
        "cellwhisperer"
    resources:
        mem_mb=100000,
        slurm=slurm_gres("medium", num_cpus=8)
    log:
        "logs/musk_zero_shot_classification_{model}_{dataset}_seed{seed}.log"
    shell: """
        LOG=`realpath {log}`
        cd {PROJECT_DIR}/modules/MUSK/benchmarks
        python3 -m clip_benchmark.cli eval \
            --pretrained_model {params.model_descriptor} \
            --dataset {wildcards.dataset} \
            --task zeroshot_classification \
            --batch_size {params.batch_size} \
            --num_workers 0 \
            --dataset_root {input.dataset_dir} \
            --seed {wildcards.seed} \
            --output {output.results} 2> $LOG
    """



# BENCHMARKS = [
#     # skincancer: 395x395 (can be seen as the zoomed-out version)
#     ("skin", "zeroshot_classification"),  # class cols: `malignancy`, `entity(,subentity)`
#     # pannuke: 256x256 is a bit more zoomed-in than skincancer
#     ("pannuke", "zeroshot_classification"),  # `caption`: combined entity and malignancy
# ]

rule musk_performance_summary:
    """
    Summarize all MUSK evaluation results
    """
    input:
        # # Zero-shot classification results
        expand(
            rules.musk_zero_shot_classification.output.results,
            dataset=["skin", "pannuke"],
            allow_missing=True
        ),

        # # Zero-shot retrieval results
        # expand(
        #     rules.musk_zero_shot_retrieval.output.results,
        #     dataset=["pathmmu_retrieval"],
        #     allow_missing=True
        # ),

        # # Image retrieval results
        # expand(
        #     rules.musk_image_retrieval.output.results,
        #     dataset=["unitopatho_retrieval"],
        #     allow_missing=True
        # ),
        # # Few-shot classification results
        # expand(
        #     rules.musk_few_shot_classification.output.results,
        #     dataset=["skin", "pannuke", "unitopatho"],
        #     k_shot=[10, 20],
        #     allow_missing=True
        # ),
        # # Linear probe results
        # expand(
        #     rules.musk_linear_probe.output.results,
        #     dataset=["skin", "pannuke", "unitopatho"],
        #     allow_missing=True
        # )
    output:
        summary=MUSK_MODEL_RESULTS / "summary" / "seed{seed}" / "musk_evaluation_summary.json"
    conda:
        "cellwhisperer"
    resources:
        mem_mb=10000,
        slurm="cpus-per-task=1"
    log:
        notebook="../logs/musk_performance_summary_{model}_seed{seed}.ipynb"
    notebook:
        "../notebooks/musk_performance_summary.py.ipynb"

rule musk_per_class_analysis:
    """
    Generate per-class analysis comparing trimodal vs bimodal models
    for pannuke and skin datasets
    """
    input:
        # Results from trimodal and bimodal_matching models for pannuke and skin
        lambda wildcards: [
            MUSK_RESULTS / "spotwhisperer_{}".format(combo) / "results" / "zeroshot_classification_{}_seed{}.json".format(dataset, wildcards.seed)
            for combo in ["cellxgene_census__archs4_geo__hest1k__quilt1m",  # trimodal
                         "cellxgene_census__archs4_geo", "hest1k", "quilt1m"]  # bimodal matching
            for dataset in ["pannuke", "skin"]
        ]
    output:
        analysis=MUSK_RESULTS / "comparison" / "per_class_analysis_seed{seed}.csv",
        plot=MUSK_RESULTS / "comparison" / "per_class_analysis_seed{seed}.pdf"
    params:
        datasets=["pannuke", "skin"] * 3,
        model_types=["trimodal", "trimodal",  "bimodal_mismatch1", "bimodal_mismatch1", "bimodal_mismatch2", "bimodal_mismatch2", "bimodal_matching", "bimodal_matching"]
    conda:
        "cellwhisperer"
    resources:
        mem_mb=20000,
        slurm="cpus-per-task=2"
    log:
        notebook="../logs/musk_per_class_analysis_seed{seed}.ipynb"
    notebook:
        "../notebooks/per_class_musk_analysis.py.ipynb"

# Main rule to generate all MUSK evaluation results
rule musk_all:
    input:
        # Performance summary
        expand(
            rules.musk_performance_summary.output.summary,
            model=["spotwhisperer_cellxgene_census__archs4_geo__hest1k__quilt1m"],
            seed=1
        ),
        # Per-class analysis
        expand(
            rules.musk_per_class_analysis.output.analysis,
            seed=1
        )
    default_target: True



# rule musk_zero_shot_retrieval:
#     """
#     Zero-shot image-text retrieval evaluation using MUSK benchmark
#     Adapted from demo.ipynb commands for image-text retrieval tasks
#     """
#     input:
#         model=PROJECT_DIR / config["paths"]["jointemb_models"] / "{model}.ckpt",
#         dataset_dir=directory(DATASET_ROOT)
#     output:
#         results=MUSK_MODEL_RESULTS / "results" / "zeroshot_retrieval_{dataset}_seed{seed}.json"
#     threads: 8
#     params:
#         batch_size=256,
#         recall_k=[1, 10, 50],
#         model_descriptor=model_descriptor_fn
#     wildcard_constraints:
#         dataset="pathmmu_retrieval",
#         seed="\\d+"
#     conda:
#         "cellwhisperer"
#     resources:
#         mem_mb=50000,
#         slurm=slurm_gres("medium", num_cpus=8)
#     log:
#         "logs/musk_zero_shot_retrieval_{model}_{dataset}_seed{seed}.log"

#     # TODO we need to to move MUSK somewhere proper (submodule would be best)
#     shell: """
#         cd {PROJECT_DIR}/modules/MUSK/benchmarks
#         python3 -m clip_benchmark.cli eval \
#             --pretrained_model {params.model_descriptor} \
#             --dataset {wildcards.dataset} \
#             --task zeroshot_retrieval \
#             --batch_size {params.batch_size} \
#             --num_workers {threads} \
#             --seed {wildcards.seed} \
#             --recall_k {params.recall_k} \
#             --dataset_root {input.dataset_dir} \
#             --output {output.results} 2> {log}
#     """

# rule musk_image_retrieval:
#     """
#     Image-to-image retrieval evaluation using MUSK benchmark
#     """
#     input:
#         model=PROJECT_DIR / config["paths"]["jointemb_models"] / "{model}.ckpt",
#         dataset_dir=directory(DATASET_ROOT)
#     output:
#         results=MUSK_MODEL_RESULTS / "results" / "image_retrieval_{dataset}_seed{seed}.json"
#     threads: 8
#     params:
#         batch_size=128,
#         model_descriptor=model_descriptor_fn
#     wildcard_constraints:
#         dataset="unitopatho_retrieval",
#         seed="\\d+"
#     conda:
#         "cellwhisperer"
#     resources:
#         mem_mb=50000,
#         slurm=slurm_gres("medium", num_cpus=8)
#     log:
#         "logs/musk_image_retrieval_{model}_{dataset}_seed{seed}.log"
#     shell: """
#         cd {PROJECT_DIR}/modules/MUSK/benchmarks
#         python3 -m clip_benchmark.cli eval \
#             --pretrained_model {params.model_descriptor} \
#             --dataset {wildcards.dataset} \
#             --task image_retrieval \
#             --batch_size {params.batch_size} \
#             --num_workers {threads} \
#             --seed {wildcards.seed} \
#             --dataset_root {input.dataset_dir} \
#             --output {output.results} 2> {log}
#     """

# rule musk_few_shot_classification:
#     """
#     Few-shot classification evaluation using linear probe
#     """
#     input:
#         model=PROJECT_DIR / config["paths"]["jointemb_models"] / "{model}.ckpt",
#         dataset_dir=directory(DATASET_ROOT)
#     output:
#         results=MUSK_MODEL_RESULTS / "results" / "fewshot_classification_{dataset}_k{k_shot}_seed{seed}.json"
#     threads: 8
#     params:
#         batch_size=256,
#         model_descriptor=model_descriptor_fn
#     wildcard_constraints:
#         dataset="skin|pannuke|unitopatho",
#         k_shot="\\d+",
#         seed="\\d+"
#     conda:
#         "cellwhisperer"
#     resources:
#         mem_mb=100000,
#         slurm=slurm_gres("large", num_cpus=8)
#     log:
#         "logs/musk_few_shot_classification_{model}_{dataset}_k{k_shot}_seed{seed}.log"
#     shell: """
#         cd {PROJECT_DIR}/modules/MUSK/benchmarks
#         python3 -m clip_benchmark.cli eval \
#             --pretrained_model {params.model_descriptor} \
#             --dataset {wildcards.dataset} \
#             --task linear_probe \
#             --batch_size {params.batch_size} \
#             --num_workers {threads} \
#             --fewshot_k {wildcards.k_shot} \
#             --seed {wildcards.seed} \
#             --dataset_root {input.dataset_dir} \
#             --output {output.results} 2> {log}
#     """

# rule musk_linear_probe:
#     """
#     Full dataset linear probe classification evaluation
#     """
#     input:
#         model=PROJECT_DIR / config["paths"]["jointemb_models"] / "{model}.ckpt",
#         dataset_dir=directory(DATASET_ROOT)
#     output:
#         results=MUSK_MODEL_RESULTS / "results" / "linear_probe_{dataset}_seed{seed}.json"
#     threads: 8
#     params:
#         batch_size=256,
#         fewshot_k=-1,  # -1 means use full dataset
#         model_descriptor=model_descriptor_fn
#     wildcard_constraints:
#         dataset="skin|pannuke|unitopatho",
#         seed="\\d+"
#     conda:
#         "cellwhisperer"
#     resources:
#         mem_mb=100000,
#         slurm=slurm_gres("large", num_cpus=8)
#     log:
#         "logs/musk_linear_probe_{model}_{dataset}_seed{seed}.log"
#     shell: """
#         cd {PROJECT_DIR}/modules/MUSK/benchmarks
#         python3 -m clip_benchmark.cli eval \
#             --pretrained_model {params.model_descriptor} \
#             --dataset {wildcards.dataset} \
#             --task linear_probe \
#             --batch_size {params.batch_size} \
#             --num_workers {threads} \
#             --fewshot_k {params.fewshot_k} \
#             --seed {wildcards.seed} \
#             --dataset_root {input.dataset_dir} \
#             --ms_aug \
#             --output {output.results} 2> {log}
#     """

