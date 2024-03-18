import json

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
assert (
    len(all_conversations_dict) == summed_length
), "Found duplicate keys in the splits"

# Convert the dictionary to a list corresponding to the required LLaVA format
conversation_list = []
for key, conversations in all_conversations_dict.items():
    renamed_conv = [
        {
            "from": {"ai": "gpt", "researcher": "human"}[conv["from"]],
            "value": conv["value"],
        }
        for conv in conversations
    ]

    conversation_list.append({"id": key, "image": key, "conversations": renamed_conv})

with open(snakemake.output.llava_stage2_dataset, "w") as f:  # type: ignore [reportUndefinedVariable]
    json.dump(conversation_list, f)
