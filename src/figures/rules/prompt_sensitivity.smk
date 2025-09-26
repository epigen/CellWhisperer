"""
#569
"""

# Carnegie stages, condensed with GPT-4o
QUERIES = {}
QUERIES['Zygote_Carnegie_Derived'] = "The embryonic stage begins with a fertilized oocyte containing pronuclei. It progresses with cell division, reducing cytoplasmic volume and forming inner and outer cell masses."  # CS1,2
QUERIES['Blastula_Carnegie_Derived'] = "The blastula stage begins with the loss of the zona pellucida, forming a free blastocyst. It progresses as the blastocyst attaches and implants into the uterine lining."  # CS3, 4, 5
QUERIES['Gastrula_Carnegie_Derived'] = "During the gastrula stage, the extraembryonic mesoderm and primitive streak form. In the neurula stage, gastrulation continues with the appearance of the notochordal process, primitive pit, and notochordal canal."  # CS6, 7, 8
QUERIES['Organogenesis_Carnegie_Derived'] = "During organogenesis, somites progressively form, neural folds fuse, and the rostral and caudal neuropores close. As development continues, limb buds, sensory structures, and facial features emerge, followed by ossification, trunk straightening, and the rounding of the head, body, and limbs."  # CS 9-23

rule generate_query_variants_download:
    """
    For reproducibility and to avoid OpenAI API calls, we provide precomputed results
    """
    input:
        HTTP.remote(f"{config['precomputing_base_url']}/misc/prompt_sensitivity_query_variants.csv", keep_local=False)[0],
    output:
        query_variants=PROJECT_DIR / "results" / "prompt_sensitivity"/ "query_variants.csv"
    run:
        import shutil
        shutil.copy(input[0], output.query_variants)

# rule generate_query_variants:
#     """
#     snakemake --draft-notebook /msc/home/mschae83/cellwhisperer_private/results/prompt_sensitivity/query_variants.csv
#     """
#     output:
#         query_variants=PROJECT_DIR / "results" / "prompt_sensitivity"/ "query_variants.csv"
#     params:
#         seed=12345,
#         temperature=0.5,
#         api_key=os.getenv("OPENAI_API_KEY"),
#         model="o1-2024-12-17",
#         system_prompt="Rewrite the provided text in 5 different variants. The length of the variants may vary slightly, but make sure that the semantics (i.e. the meaning of the generated variant text) remains close to the initially provided text. Return the variants as a JSON-formatted list of strings (key=\"variants\").",
#         queries=list(QUERIES.values())
#     conda:
#         "cellwhisperer"
#     log:
#         progress="../logs/generate_query_variants.log",
#         notebook="../logs/generate_query_variants.ipynb"
#     notebook:
#         "../notebooks/generate_query_variants.py.ipynb"

rule compute_text_embeddings:
    """
    Embed the query variants using the cellwhisperer text embedding model 
    """
    input:
        query_variants=rules.generate_query_variants_download.output.query_variants,
        model=PROJECT_DIR / config["paths"]["jointemb_models"] / "{model}.ckpt",
    output:
        text_embeddings=PROJECT_DIR / "results" / "prompt_sensitivity" / "{model}" / "text_embeddings.pt",
    conda:
        "cellwhisperer"
    resources:
        mem_mb=40000,
        slurm=slurm_gres("large", num_gpus=1, num_cpus=10)
    log:
        progress="../logs/compute_text_embeddings_{model}.log",
        notebook="../logs/compute_text_embeddings_{model}.ipynb"
    notebook:
        "../notebooks/compute_text_embeddings.py.ipynb"

rule plot_query_variants:
    """
    Plot the query variants
    """
    input:
        query_variants=rules.generate_query_variants_download.output.query_variants,
        text_embeddings=rules.compute_text_embeddings.output.text_embeddings,
    output:
        plot=PROJECT_DIR / "results" / "plots" / "prompt_sensitivity" / "{model}" /  "query_variants.svg"
    conda:
        "cellwhisperer"
    log:
        notebook="../logs/plot_query_variants_{model}.ipynb"
    notebook:
        "../notebooks/plot_query_variants.py.ipynb"

rule plot_query_variant_cell_matching:
    """
    """

    input:
        query_variants=rules.generate_query_variants_download.output.query_variants,
        text_embeddings=rules.compute_text_embeddings.output.text_embeddings,
        dataset=PROJECT_DIR / config["paths"]["model_processed_dataset"].format(dataset="development", model="{model}")
    output:
        plot=PROJECT_DIR / "results" / "plots" / "prompt_sensitivity" / "{model}" / "query_variant_cell_matching.svg"
    params:
        carnegie_stages=QUERIES,
    resources:
        mem_mb=100000,
        # slurm=f"cpus-per-task=5 gres=gpu:a100:1 qos=a100 partition=gpu"
        slurm=f"cpus-per-task=5 gres=gpu:a100-sxm4-80gb:1 qos=a100-sxm4-80gb partition=gpu"
    conda:
        "cellwhisperer"
    log:
        notebook="../logs/compute_text_embeddings_{model}.ipynb"
    notebook:
        "../notebooks/plot_query_variant_cell_matching.py.ipynb"
