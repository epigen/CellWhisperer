import json
import logging
import random
from pathlib import Path
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filename=snakemake.log[0],  # type: ignore [reportUndefinedVariable]
    filemode="w",
)

splits = []
for split_fn in snakemake.input.json_splits:  # type: ignore [reportUndefinedVariable]
    with open(split_fn) as f:
        splits.append(json.load(f))

# Concatenate the splits into a single JSON
all_conversations_dict = {}
summed_length = 0
for split in splits:
    summed_length += len(split)
    all_conversations_dict.update(split)
# Make sure that the keys of the splits are unique
if len(all_conversations_dict) != summed_length:
    logging.warning("Found duplicate keys in the splits")

# Add the complex and detailed questions
num_erroneous_jsons = 0
for special_conversation_fn in snakemake.input.complex_conversations + snakemake.input.detailed_conversations:  # type: ignore [reportUndefinedVariable]
    conv_text = Path(special_conversation_fn).read_text()
    if conv_text.startswith("```"):
        logging.warning(
            f"Found code block in conversation file {special_conversation_fn}. stripping"
        )
        conv_text = conv_text.replace("```json", "").replace("```", "")

    try:
        complex_conversation = json.loads(conv_text)
    except json.JSONDecodeError as e:
        logging.warning(
            f"Error while loading conversation file {special_conversation_fn}: {e}"
        )
        num_erroneous_jsons += 1
        continue

    sample_id = Path(special_conversation_fn).stem

    # Remove the initial "contemplation" step
    if "contemplation" in complex_conversation[0]:
        complex_conversation = complex_conversation[1:]
    else:
        logging.warning(
            "The first step should be contemplation for sample id {sample_id}"
        )

    # NOTE Apparently we accidentally generated conversations for all 100k conversations. We overwrite them here with the complex ones
    # if sample_id in all_conversations_dict:
    #     logging.warning("Duplicate sample id found")
    all_conversations_dict[sample_id] = complex_conversation
if num_erroneous_jsons > snakemake.params.accept_num_erroneous_jsons:
    raise ValueError(f"Too many erroneous conversations ({num_erroneous_jsons})")


# Add the description questions from dataset1
with open(snakemake.input.stage1_train_set) as f:  # type: ignore [reportUndefinedVariable]
    stage1_train_set = json.load(f)
    stage1_train_set = {obj["id"]: obj for obj in stage1_train_set}


with open(snakemake.input.stage1_test_set) as f:  # type: ignore [reportUndefinedVariable]
    stage1_test_set = json.load(f)
    stage1_test_set = {obj["id"]: obj for obj in stage1_test_set}


# Choose the stage1 samples to include
stage1_candidates = list(
    (set(stage1_train_set.keys()) | set(stage1_test_set.keys()))
    - set(all_conversations_dict.keys())
)
# Load transcriptome and annotation weights and convert to dict
transcriptome_weights = [
    np.load(fn, allow_pickle=True)  # type: ignore [reportUndefinedVariable]
    for fn in snakemake.input.transcriptome_weights
]
transcriptome_weights = {
    sample_id: weight
    for d in transcriptome_weights
    for sample_id, weight in zip(d["orig_ids"], d["weight"])
}
annotation_weights = [
    np.load(fn, allow_pickle=True)  # type: ignore [reportUndefinedVariable]
    for fn in snakemake.input.annotation_weights
]
annotation_weights = {
    sample_id: weight
    for d in annotation_weights
    for sample_id, weight in zip(d["orig_ids"], d["weight"])
}

# Filter the weights to only include the stage1 candidates
stage1_candidate_weights = np.array(
    [
        (transcriptome_weights[sample_id] + annotation_weights[sample_id]) / 2
        for sample_id in stage1_candidates
    ]
)

np.random.seed(snakemake.params.seed)  # type: ignore [reportUndefinedVariable]
stage1_samples = np.random.choice(
    stage1_candidates,
    size=snakemake.params.num_stage1_samples,  # type: ignore [reportUndefinedVariable]
    p=stage1_candidate_weights / (stage1_candidate_weights).sum(),
    replace=False,
)

# Add the sampled stage1 samples to the dataset
for sample_id in stage1_samples:
    try:
        all_conversations_dict[sample_id] = stage1_train_set[sample_id]["conversations"]
    except KeyError:
        all_conversations_dict[sample_id] = stage1_test_set[sample_id]["conversations"]

# Convert the dictionary to a list corresponding to the required LLaVA format
training_list = []
test_list = []

for key, conversations in all_conversations_dict.items():
    if (
        len(conversations) == 0
    ):  # Mistral-based generation seems to have produced empty conversations
        continue
    try:
        renamed_conv = [
            {
                "from": {
                    "ai": "gpt",
                    "gpt": "gpt",
                    "researcher": "human",
                    "human": "human",
                }[conv["from"]],
                "value": conv["value"],
            }
            for conv in conversations
        ]
    except TypeError as e:
        logging.warning(
            f"Error while processing sample {key}. The error was: {e}. The content: {conversations}. Skipping sample."
        )
        continue
    # the stage1 dataset has already been added
    if snakemake.params.transcriptome_tag not in renamed_conv[0]["value"]:  # type: ignore [reportUndefinedVariable]
        if bool(random.getrandbits(1)):
            renamed_conv[0]["value"] = f"{snakemake.params.transcriptome_tag}\n{renamed_conv[0]['value']}"  # type: ignore [reportUndefinedVariable]
        else:
            renamed_conv[0]["value"] = f"{renamed_conv[0]['value']}\n{snakemake.params.transcriptome_tag}"  # type: ignore [reportUndefinedVariable]

    obj = {"id": key, "image": key, "conversations": renamed_conv}

    if key in snakemake.params.test_ids:  # type: ignore [reportUndefinedVariable]
        test_list.append(obj)
    else:
        training_list.append(obj)

with open(snakemake.output.llava_stage2_dataset, "w") as f:  # type: ignore [reportUndefinedVariable]
    json.dump(training_list, f)

with open(snakemake.output.evaluation_dataset, "w") as f:  # type: ignore [reportUndefinedVariable]
    json.dump(test_list, f)
