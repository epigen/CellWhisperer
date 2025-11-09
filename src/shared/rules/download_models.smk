BASE_URL = "https://medical-epigenomics.org/papers/schaefer2025cellwhisperer/data"

rule download_mixtral:
    input:
        HTTP.remote("https://huggingface.co/TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF/resolve/main/mixtral-8x7b-instruct-v0.1.Q5_K_M.gguf?download=true", keep_local=False)[0]
    output:
        PROJECT_DIR / config["model_name_path_map"]["mixtral"]
    shell: """
        cp {input} {output}
    """

rule download_geneformer:
    """
    Explicitly download geneformer files (We dropped git-lfs in our geneformer fork, which contained the model files.)
    """
    input:
        HTTP.remote("https://huggingface.co/ctheodoris/Geneformer/resolve/9d41e7053af8a702003d99305cee01cd34b62ab7/geneformer-12L-30M/config.json?download=true", keep_local=False),
        HTTP.remote("https://huggingface.co/ctheodoris/Geneformer/resolve/9d41e7053af8a702003d99305cee01cd34b62ab7/geneformer-12L-30M/pytorch_model.bin?download=true", keep_local=False),
        HTTP.remote("https://huggingface.co/ctheodoris/Geneformer/resolve/9d41e7053af8a702003d99305cee01cd34b62ab7/geneformer-12L-30M/training_args.bin?download=true", keep_local=False)
    output:
        PROJECT_DIR / config["model_name_path_map"]["geneformer"] / "config.json",
        PROJECT_DIR / config["model_name_path_map"]["geneformer"] / "pytorch_model.bin",
        PROJECT_DIR / config["model_name_path_map"]["geneformer"] / "training_args.bin",
        directory(PROJECT_DIR / config["model_name_path_map"]["geneformer"] ),
    run:
        import shutil
        for fin, fout in zip(input, output):
            shutil.copy(fin, fout)

rule download_scgpt:
    output:
        folder=directory(PROJECT_DIR / config["model_name_path_map"]["scgpt"])
    conda:
        "../../../envs/gdown.yaml"
    shell:"""
        mkdir {output[0]}
        cd {output.folder}/..
        gdown --folder --id 1oWh_-ZRdhtoGQ2Fw24HP41FgLoomVo-y
    """


rule download_uce:
    output:
        PROJECT_DIR / config["model_name_path_map"]["uce4layer"],
        PROJECT_DIR / config["model_name_path_map"]["uce"],
        directory(PROJECT_DIR / config["uce_paths"]["protein_embeddings_dir"]),
        PROJECT_DIR / config["uce_paths"]["offset_pkl_path"],
        PROJECT_DIR / config["uce_paths"]["spec_chrom_csv_path"],
        PROJECT_DIR / config["uce_paths"]["tokens"],
    shell:
        """
        urls=(
            "https://figshare.com/ndownloader/files/42706576"
            "https://figshare.com/ndownloader/files/43423236"
            "https://figshare.com/ndownloader/files/42715213"
            "https://figshare.com/ndownloader/files/42706555"
            "https://figshare.com/ndownloader/files/42706558"
            "https://figshare.com/ndownloader/files/42706585"
        )
        outs=({output})
        i=0
        for url in "${{urls[@]}}"; do
            wget --header="User-Agent: Mozilla/5.0" -O "${{outs[$i]}}" "$url"
            i=$((i+1))
        done

        # unpack protein embeddings
        tar -xzf {output[2]} -C $(dirname {output[2]})
        """

rule download_llama33:
    output:
        directory(PROJECT_DIR / config["model_name_path_map"]["llama33"])
    shell: """
        echo "You'll need a huggingface token to download llama 3.3. You may also download it manually into {output}"
        git lfs install
        git clone https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct {output}
    """

rule download_mistral:
    output:
        directory(PROJECT_DIR / config["model_name_path_map"]["mistral"])
    shell: """
        echo "You'll need a huggingface token to download mistral 7b. You may also download it manually into {output}"
        git lfs install
        git clone https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2 {output}
    """


rule download_cellwhisperer_embedding_model:
    input:
        HTTP.remote(f"{BASE_URL}/models/{{cw_model}}.ckpt")[0]
    output:
        PROJECT_DIR / config["paths"]["jointemb_models"] / "{cw_model}.ckpt"
    shell: """
        cp {input} {output}
        """

rule download_cellwhisperer_llms:
    """
    """
    input:
        HTTP.remote(f"{BASE_URL}/models/{{base_model}}__{{model}}__{{llava_dataset}}.tar.gz")
    output:
        directory(PROJECT_DIR / config["paths"]["llava"]["finetuned_model_dir"])
    shell: """
        mkdir {output}
        tar -xzvf {input} -C {output} --no-same-owner
    """


