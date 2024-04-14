import os
from openai import OpenAI
from pathlib import Path
import json
import pandas as pd
import logging
import re


import os
from pathlib import Path
import subprocess
from typing import Optional


# setup snakemake logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(snakemake.log[0]), logging.StreamHandler()],  # type: ignore [reportUndefinedVariable]
)


def load_openai_api(api_key: Optional[str] = None):
    """Make sure OpenAI API key is available. If not, then load it

    Args:
        api_key: The OpenAI API key to use. If None, then try to load it from the environment variable OPENAI_API_KEY or from the password store.
        model: The model to use. Defaults to "gpt-4". alternatively use gpt-3.5-turbo for faster and cheaper exploration
    """
    try:
        return os.environ["OPENAI_API_KEY"]
    except KeyError:
        if api_key is None:

            def get_password_from_pass_store(key: str) -> str:
                # Run the pass command to get the password for the specified key
                password = subprocess.run(["pass", key], capture_output=True, text=True)

                # Check if the command execution was successful
                if password.returncode != 0:
                    raise RuntimeError(f"Failed to get password for key: {key}")

                # Remove trailing newline character and return the password
                return password.stdout.rstrip("\n")

            key = "openai.com/meduni_my_api_key"  # Moritz' setup

            return get_password_from_pass_store(key)


with open(snakemake.input.processed_annotations) as f:  # type: ignore [reportUndefinedVariable]
    annotations = json.load(f)

    annotations = annotations[snakemake.wildcards.sample_id]  # type: ignore [reportUndefinedVariable]
    logging.info(
        f"Loaded annotations for {snakemake.wildcards.sample_id}. Choosing #{snakemake.params.annotation_replicate} of:\n{annotations}"  # type: ignore [reportUndefinedVariable]
    )
    annotation = annotations[snakemake.params.annotation_replicate]  # type: ignore [reportUndefinedVariable]


gsva_gene_sets = (
    pd.read_parquet(snakemake.input.gsva)
    .set_index("Unnamed: 0")
    .drop(columns=["library"])
)
top_gene_sets = (
    gsva_gene_sets[snakemake.wildcards.sample_id]  # type: ignore [reportUndefinedVariable]
    .sort_values(ascending=False)
    .index.to_list()[: snakemake.params.top_n_gene_sets]  # type: ignore [reportUndefinedVariable]
)
top_gene_sets = [re.sub(r"\(.*\)", "", gs) for gs in top_gene_sets]


top_genes = (
    pd.read_parquet(snakemake.input.top_genes)
    .loc[snakemake.wildcards.sample_id]
    .dropna()
    .to_list()[: snakemake.params.top_n_genes]  # type: ignore [reportUndefinedVariable]
)

request_template = Path(snakemake.input.request_template).read_text()

# Instruction
request = [
    {"role": "system", "content": Path(snakemake.input.system_message).read_text()}
]

# Few-shot prompts
for fs_request_fn, fs_response_fn in zip(
    snakemake.input.few_shot_prompts, snakemake.input.few_shot_responses
):
    with open(fs_request_fn) as f:
        fs_request = json.load(f)

    request.append({"role": "user", "content": request_template.format(**fs_request)})
    request.append({"role": "assistant", "content": Path(fs_response_fn).read_text()})

# Main request
request.append(
    {
        "role": "user",
        "content": request_template.format(
            annotation=annotation, top_gene_sets=top_gene_sets, top_genes=top_genes
        ),
    }
)

# pretty print full request (to log)
logging.info("Full request:")
logging.info(json.dumps(request, indent=2))

client = OpenAI()

chat_completion = client.chat.completions.create(
    messages=request,
    model="gpt-4-turbo-preview",
    #     response_format={"type": "json_object"},  #
    max_tokens=2048,
    temperature=snakemake.params.temperature,  # type: ignore [reportUndefinedVariable]
)
response = chat_completion.choices[0].message.content

logging.info("Response:")
logging.info(json.dumps(response, indent=2))

Path(snakemake.output.processed_annotation).write_text(response)  # type: ignore [reportUndefinedVariable]
