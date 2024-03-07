from llama_cpp_cuda_tensorcores import (
    Llama,
)  # same speed as llama_cpp_cuda, but with tensor cores
import json
import yaml
import pandas as pd
from pathlib import Path
import logging


prompt_template = Path(snakemake.input.instruction).read_text()
with open(snakemake.input.yaml_split) as f:
    yaml_split = yaml.load(f, Loader=yaml.FullLoader)

# first sample, set temperature to 0.2. for the others set it to 0.7
temperature = 0.2 if int(snakemake.wildcards.replicate) == 0 else 0.7


# load the model
llm = Llama(
    model_path=snakemake.input.model,
    n_ctx=32000,  # The max sequence length to use - note that longer sequence lengths require much more resources
    n_threads=25,  # The number of CPU threads to use
    n_gpu_layers=86,  # High enough number to load the full model
)

few_shot_block = []

few_shot_prompts = sorted(Path(snakemake.input.few_shot_samples).glob("*.json"))

for prompt_file in few_shot_prompts:
    response_file = prompt_file.parent / f"{prompt_file.stem}.txt"
    if not response_file.exists():
        continue

    few_shot_block += f"Structured: {json.dumps(json.loads(Path(prompt_file).read_text()))}\nNatural Language: {Path(response_file).read_text()}"


results = []

for key, sample in yaml_split.items():
    # NOTE: I could use the template few shot prompting: <s>[INST] Instruction, example 1 [/INST] Model answer 1 [INST] example 2 [/INST] Model answer 2</s>[INST] query [/INST]. For this. use the chat API: https://www.reddit.com/r/Oobabooga/comments/17yxl42/comment/k9wmjgt/?utm_source=share&utm_medium=web3x&utm_name=web3xcss&utm_term=1&utm_content=share_button (mixtral supports (only) user and assistant roles)
    serialized_sample = json.dumps(sample)
    # trim to reasonable length (20000 * 4 characters per token), just in case
    if len(serialized_sample) > 80000:
        logging.warning(f"Sample {key} too long, trimming to 80000 characters")
        serialized_sample = serialized_sample[:80000]

    for i in range(10):
        output = llm(
            "<s>[INST] "
            + prompt_template.format(
                few_shot_block="\n".join(few_shot_block), query=serialized_sample
            )
            + " [/INST] ",
            max_tokens=256,  # for training, we only use a max of 128. observe whether this matches..
            stop=["</s>"],  # stop token for Mixtral
            logit_bias={
                llm.tokenizer().encode("\n")[-1]: float("-inf")
            },  # Prevent newlines
            echo=False,  # Whether to echo the prompt
            seed=i + hash(f"{key}{snakemake.wildcards.replicate}") % 2**30,
            temperature=temperature + i * 0.1,
            top_p=0.9,
            top_k=50,
        )

        result = output["choices"][0]["text"].strip()
        # Only accept results that are not too short
        if len(result) > 20:
            break
        else:
            logging.warning(
                f"Too short response for sample {key}: {result}, retrying {i}"
            )
    else:
        result = "No information for this sample available"
        logging.warning(
            f"Prompt template continuously failed: \n{serialized_sample}\n{sample}"
        )

    results.append(
        {
            "sample_id": key,
            "replicate": int(snakemake.wildcards.replicate),
            "annotation": result,
        }
    )
pd.DataFrame(results, columns=["sample_id", "replicate", "annotation"]).to_csv(
    snakemake.output.processed_annotation, index=False
)
