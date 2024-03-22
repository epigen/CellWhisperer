import json
import logging
import random

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

# Add the description questions from dataset1
with open(snakemake.input.stage1_train_set) as f:  # type: ignore [reportUndefinedVariable]
    stage1_train_set = json.load(f)
    stage1_train_set = {obj["id"]: obj for obj in stage1_train_set}

with open(snakemake.input.stage1_test_set) as f:  # type: ignore [reportUndefinedVariable]
    stage1_test_set = json.load(f)
    stage1_test_set = {obj["id"]: obj for obj in stage1_test_set}

# Convert the dictionary to a list corresponding to the required LLaVA format
training_list = []
test_list = []
for key, conversations in all_conversations_dict.items():
    renamed_conv = [
        {
            "from": {"ai": "gpt", "researcher": "human"}[conv["from"]],
            "value": conv["value"],
        }
        for conv in conversations
    ]

    if bool(random.getrandbits(1)):
        renamed_conv[0]["value"] = f"{snakemake.params.transcriptome_tag}\n{renamed_conv[0]['value']}"  # type: ignore [reportUndefinedVariable]
    else:
        renamed_conv[0]["value"] = f"{renamed_conv[0]['value']}\n{snakemake.params.transcriptome_tag}"  # type: ignore [reportUndefinedVariable]

    obj = {"id": key, "image": key, "conversations": renamed_conv}

    if key in snakemake.params.test_ids:  # type: ignore [reportUndefinedVariable]
        test_list.append(obj)
        test_list.append(stage1_test_set[key])
    else:
        training_list.append(obj)
        training_list.append(stage1_train_set[key])

with open(snakemake.output.llava_stage2_dataset, "w") as f:  # type: ignore [reportUndefinedVariable]
    json.dump(training_list, f)

with open(snakemake.output.evaluation_dataset, "w") as f:  # type: ignore [reportUndefinedVariable]
    json.dump(test_list, f)
