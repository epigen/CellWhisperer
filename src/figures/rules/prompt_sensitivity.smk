
"""
#569
"""

QUERIES = {
    "Carnegie stage 01": "Embryonic stage defined by a fertilized oocyte and presence of pronuclei.",
    "Carnegie stage 02": "Embryonic stage during which cell division occurs with reduction in cytoplasmic volume, and formation of inner and outer cell mass.",
    "Carnegie stage 03": "Blastula stage with the loss of the zona pellucida and the definition of a free blastocyst.",
    "Carnegie stage 04": "Blastula stage during which the blastocyst becomes attached.",
    "Carnegie stage 05": "Blastula stage during which implantation occurs.",
    "Carnegie stage 06": "Gastrula stage during which the extraembryonic mesoderm and primitive streak appear.",
    "Carnegie stage 07": "Neurula stage during which the notochordal process appears, and gastrulation is continuing.",
    "Carnegie stage 08": "Neurula stage during which the primitive pit and the notochordal canal appear.",
    "Carnegie stage 09": "Organogenesis stage during which somites 1-3 appear, and neural folds, cardiac primordium, and head fold are present.",
    "Carnegie stage 10": "Organogenesis stage during which somites 4-12 appear, and the neural fold fuses.",
    "Carnegie stage 11": "Organogenesis stage during which somites 13-20 appear, and the rostral neuropore closes.",
    "Carnegie stage 12": "Organogenesis stage during which somites 21-29 appear, and the caudal neuropore closes.",
    "Carnegie stage 13": "Organogenesis developmental stage during which somite 30 appears, and leg buds, lens placode, pharyngeal arches are present.",
    "Carnegie stage 14": "Organogenesis stage during which lens pit and optic cup appear.",
    "Carnegie stage 15": "Organogenesis stage during which lens vesicle, nasal pit, and hand plate appear.",
    "Carnegie stage 16": "Organogenesis stage during which nasal pits moved ventrally, and features are auricular hillocks, and foot plate.",
    "Carnegie stage 17": "Organogenesis stage during which finger rays become visible.",
    "Carnegie stage 18": "Organogenesis stage during which ossification commences.",
    "Carnegie stage 19": "Organogenesis stage during which the straightening of the trunk starts.",
    "Carnegie stage 20": "Organogenesis stage during which upper limbs are longer and bent at elbow.",
    "Carnegie stage 21": "Organogenesis stage during which hands and feet turned inward.",
    "Carnegie stage 22": "Organogenesis stage during which eyelids and external ears appear.",
    "Carnegie stage 23": "Organogenesis stage during which the head, the body, and the limbs are rounded structures.",
    # "Carnegie stage 05a": "Blastula stage during which implantation occurs and defined by a solid trophoblast.",  # NOTE adjusted
    # "Carnegie stage 05b": "Blastula stage during which implantation occurs and defined by a trophoblastic lacunae.",  # NOTE adjusted
    # "Carnegie stage 05c": "Blastula stage during which implantation occurs and defined by a lacunar vascular circle.",  # NOTE adjusted
    # "Carnegie stage 06a": "Gastrula stage during which the extraembryonic mesoderm and primitive streak as well as the chorionic villi appear.",  # NOTE adjusted
    # "Carnegie stage 06b": "Gastrula stage during which the extraembryonic mesoderm and primitive streak as well as the primitive streak appear."  # NOTE adjusted
}

rule generate_query_variants:
    """
    snakemake --draft-notebook /msc/home/mschae83/cellwhisperer_private/results/prompt_sensitivity/query_variants.csv
    """
    output:
        query_variants=PROJECT_DIR / "results" / "prompt_sensitivity"/ "query_variants.csv"
    params:
        seed=12345,
        temperature=0.5,
        api_key=os.getenv("OPENAI_API_KEY"),
        model="o1-2024-12-17",
        system_prompt="Rewrite the provided text in 5 different variants. The length of the variants may vary slightly, but make sure that the semantics (i.e. the meaning of the generated variant text) remains close to the initially provided text. Return the variants as a JSON-formatted list of strings (key=\"variants\").",
        queries=list(QUERIES.values())
    conda:
        "cellwhisperer"
    log:
        progress="../logs/generate_query_variants.log",
        notebook="../logs/generate_query_variants.ipynb"
    notebook:
        "../notebooks/generate_query_variants.py.ipynb"

rule compute_text_embeddings:
    """
    Embed the query variants using the cellwhisperer text embedding model 
    """
    input:
        query_variants=rules.generate_query_variants.output.query_variants,
        model=PROJECT_DIR / config["paths"]["jointemb_models"] / "{model}.ckpt",
    output:
        text_embeddings=PROJECT_DIR / "results" / "prompt_sensitivity" / "{model}" / "text_embeddings.pt",
    conda:
        "cellwhisperer"
    resources:
        mem_mb=100000,
        slurm=f"cpus-per-task=5 gres=gpu:a100:1 qos=a100 partition=gpu"
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
        query_variants=rules.generate_query_variants.output.query_variants,
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
        query_variants=rules.generate_query_variants.output.query_variants,
        text_embeddings=rules.compute_text_embeddings.output.text_embeddings,
        dataset=PROJECT_DIR / config["paths"]["model_processed_dataset"].format(dataset="development3s", model="{model}")
    output:
        plot=PROJECT_DIR / "results" / "plots" / "prompt_sensitivity" / "{model}" / "query_variant_cell_matching.svg"
    params:
        carnegie_stages=QUERIES,
    resources:
        mem_mb=100000,
        slurm=f"cpus-per-task=5 gres=gpu:a100:1 qos=a100 partition=gpu"
    conda:
        "cellwhisperer"
    log:
        notebook="../logs/compute_text_embeddings_{model}.ipynb"
    notebook:
        "../notebooks/plot_query_variant_cell_matching.py.ipynb"
