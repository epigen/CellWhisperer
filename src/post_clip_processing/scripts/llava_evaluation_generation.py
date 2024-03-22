import torch
import re

input_ids = (
    tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
    .unsqueeze(0)
    .to(device=model.device)
)
images = torch.tensor(
    eval_point["transcriptome_embedding"], device=model.device, dtype=torch.float16
).unsqueeze(0)

if len(images) != prompt.count(DEFAULT_IMAGE_TOKEN):
    raise ValueError(
        "Number of images does not match number of <image> tokens in prompt"
    )


# Process a single sample

eval_point = eval_set[0]

# For the start, we only take the first question. Later, more can be planned

qs = eval_point["conversations"][0]["value"]


if model.config.mm_use_im_start_end:
    qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
else:
    qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    conv = conv_templates["mistral_instruct"].copy()  # same as provided for fine-tuning
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    prompt
# can delete
# replace_token = DEFAULT_IMAGE_TOKEN
# if getattr(model.config, 'mm_use_im_start_end', False):
#     replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
# prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)


# TODO use the 'num-token' information in `mm_projector_type`

num_tokens = int(
    re.match(r"^mlp(\d+)x_(\d+)t_gelu$", model.config.mm_projector_type).group(2)
)
num_image_tokens = prompt.count(replace_token) * num_tokens
image_sizes = None

image_args = {"images": images, "image_sizes": image_sizes}

# load "images" (unnecessary)
processed_transcriptomes = np.load(snakemake.input.image_data, allow_pickle=True)

for sample_id, transcriptome_embed in zip(
    processed_transcriptomes["orig_ids"],
    processed_transcriptomes["transcriptome_embeds"],
):
    for eval_point in eval_set:
        if eval_point["id"] == sample_id:
            eval_point["transcriptome_embedding"] = transcriptome_embed
            break


temperature = 0.0
top_p = 1.0
max_context_length = 2048
max_new_tokens = 256  # depending on whether we go question by question
stop_str = "</s>"
do_sample = True if temperature > 0.001 else False
max_new_tokens = min(
    max_new_tokens, max_context_length - input_ids.shape[-1] - num_image_tokens
)
# keywords = [stop_str]  # maybe enable
# stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

with torch.inference_mode():
    output_ids = model.generate(
        input_ids=input_ids,
        labels=input_ids,
        use_cache=True,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        max_content_length=max_content_length,
        **image_args
    )
