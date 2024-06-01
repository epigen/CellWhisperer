from snakemake.remote.HTTP import RemoteProvider as HTTPRemoteProvider
HTTP = HTTPRemoteProvider()

BASE_URL = "https://medical-epigenomics.org/papers/schaefer2024/data"
CLIP_MODEL_FN = config["model_name_path_map"]["cellwhisperer"] + ".ckpt"

rule download_geneformer:
    """
    Explicitly download geneformer files (We dropped git-lfs in our geneformer fork, which contained the model files.)
    """
    input:
        HTTP.remote("https://huggingface.co/ctheodoris/Geneformer/resolve/main/geneformer-12L-30M/config.json?download=true", keep_local=False),
        HTTP.remote("https://huggingface.co/ctheodoris/Geneformer/resolve/main/geneformer-12L-30M/pytorch_model.bin?download=true", keep_local=False),
        HTTP.remote("https://huggingface.co/ctheodoris/Geneformer/resolve/main/geneformer-12L-30M/training_args.bin?download=true", keep_local=False)
    output:
        PROJECT_DIR / "resources" / "geneformer-12L-30M" / "config.json",
        PROJECT_DIR / "resources" / "geneformer-12L-30M" / "pytorch_model.bin",
        PROJECT_DIR / "resources" / "geneformer-12L-30M" / "training_args.bin",
    run:
        import shutil
        for fin, fout in zip(input, output):
            shutil.copy(fin, fout)

rule download_cellwhisperer_embedding_model:
    input:
        HTTP.remote(f"{BASE_URL}/models/{CLIP_MODEL_FN}")[0]
    output:
        PROJECT_DIR / config["paths"]["jointemb_models"] / CLIP_MODEL_FN
    shell: """
        cp {input} {output}
        """


rule download_cellwhisperer_llm:
    """
    """
    input:
        HTTP.remote(f"{BASE_URL}/models/Mistral-7B-Instruct-v0.2__cellwhisperer_clip_v1.tar.gz")
    output:
        directory(PROJECT_DIR / config["paths"]["llava"]["finetuned_model_dir"])
    shell: """
        mkdir {output}
        tar -xzvf {input} -C {output}
    """
