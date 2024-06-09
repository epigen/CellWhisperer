from llama_cpp import (
    Llama,
    LlamaGrammar,
)
import json
import yaml
import pandas as pd
from pathlib import Path
import logging


prompt_template = Path(snakemake.input.instruction).read_text()  # type: ignore [reportUndefinedVariable]


with open(snakemake.input.yaml_split) as f:
    yaml_split = yaml.load(f, Loader=yaml.FullLoader)

# first sample, set temperature to 0.2. for the others set it to 0.7
temperature = 0.2 if int(snakemake.wildcards.replicate) == 0 else 0.7

# load the model
llm = Llama(
    model_path=snakemake.input.model,
    n_ctx=32000,  # The max sequence length to use - note that longer sequence lengths require much more resources
    n_threads=5,  # The number of CPU threads to use
    n_threads_batch=25,
    n_gpu_layers=86,  # High enough number to load the full model
)

few_shot_block = []


def build_example(data, response_file=None):
    sample_data = {k: v for k, v in data.items() if k not in snakemake.params.study_specific_fields}  # type: ignore [reportUndefinedVariable]
    study_data = {k: v for k, v in data.items() if k in snakemake.params.study_specific_fields}  # type: ignore [reportUndefinedVariable]
    example = (
        f"Study Information: {json.dumps(study_data)}\n"
        f"Sample Information: {json.dumps(sample_data)}"
    )
    if response_file:
        example += f"\nResponse: {Path(response_file).read_text()}\n"

    return example


for prompt_file, response_file in zip(
    snakemake.input.few_shot_prompts, snakemake.input.few_shot_responses
):
    data = json.loads(Path(prompt_file).read_text())
    few_shot_block.append(build_example(data, response_file))

results = []
requests = []

for key, sample in yaml_split.items():
    # NOTE: I could use the template few shot prompting: <s>[INST] Instruction, example 1 [/INST] Model answer 1 [INST] example 2 [/INST] Model answer 2</s>[INST] query [/INST]. For this. use the chat API: https://www.reddit.com/r/Oobabooga/comments/17yxl42/comment/k9wmjgt/?utm_source=share&utm_medium=web3x&utm_name=web3xcss&utm_term=1&utm_content=share_button (mixtral supports (only) user and assistant roles)

    serialized_sample = build_example(sample)
    # trim to reasonable length (20000 * 4 characters per token), just in case (to avoid OOMs and other errors)
    if len(serialized_sample) > 80000:
        logging.warning(f"Sample {key} too long, trimming to 80000 characters")
        serialized_sample = serialized_sample[:80000]
    prompt = prompt_template.format(
        few_shot_block="\n".join(few_shot_block), query=serialized_sample
    )
    requests.append(prompt)

    for i in range(10):
        output = llm(
            f"<s>[INST] {prompt} [/INST] ",
            max_tokens=1024,  # for training, we only use a max of 128. observe whether this matches..
            stop=["</s>"],  # stop token for Mixtral
            # logit_bias={
            #     llm.tokenizer().encode("\n")[-1]: float("-inf")
            # },  # Prevent newlines
            echo=False,  # don't echo the prompt as part of the response
            seed=hash(f"{key}{snakemake.wildcards.replicate}{i}") % 2**30,
            grammar=LlamaGrammar.from_json_schema(
                Path(snakemake.input.json_schema).read_text()
            ),
            temperature=temperature + i * 0.2,
            top_p=0.9 + (i * 0.01),
            top_k=50 + (i * 20),
        )

        result_serial = output["choices"][0]["text"].strip()
        logging.debug(result_serial)
        try:
            result = json.loads(result_serial)["2. Final Response"]
        except (json.JSONDecodeError, KeyError) as e:
            logging.warning(f"Failed to decode JSON: {e}. Retrying {i}")
            continue

        # Only accept results that are not too short
        if len(result) > 20:
            break
        else:
            logging.warning(
                f"Too short response for sample {key}: {result}. Retrying {i}"
            )
    else:
        result = "No information available for this sample"
        logging.warning(
            f"Prompt template continuously failed: \n{serialized_sample}\n{sample}\n{result_serial}"
        )

    results.append(
        {
            "sample_id": key,
            "replicate": int(snakemake.wildcards.replicate),  # type: ignore [reportUndefinedVariable]
            "annotation": result,
        }
    )
pd.DataFrame(results, columns=["sample_id", "replicate", "annotation"]).to_csv(
    snakemake.output.processed_annotation, index=False
)
with open(snakemake.output.requests, "w") as f:
    yaml.dump(requests, f)
