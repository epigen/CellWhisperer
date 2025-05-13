import json
import anndata
import copy
import pandas as pd
import tqdm

# from snakemake.io import SnakemakeNamedlist

adata = anndata.read_h5ad(snakemake.input.annotations_cellxgene_census, backed="r")

celltypes = (
    adata.obs["author_cell_type"]
    .astype(str)
    .where(adata.obs["author_cell_type"].notna(), adata.obs["cell_type"])
)


# if isinstance(snakemake.input.top_genes, SnakemakeNamedlist):
#     top_genes = pd.concat(
#         [
#             pd.read_parquet(fn).iloc[:, : snakemake.params.top_n_genes]
#             for fn in snakemake.input.top_genes
#         ]
#     )
# else:
top_genes = pd.read_parquet(snakemake.input.top_genes).iloc[:, : snakemake.params.top_n_genes]  # type: ignore [reportUndefinedVariable]
top_genes.head()

# Generate conversations
conversations = []
for cell_id, celltype in tqdm.tqdm(celltypes.iteritems()):
    obj = {
        "id": cell_id,
        "conversations": [
            {
                "from": "human",
                "value": snakemake.params.question,
            },
            {
                "from": "gpt",
                "value": snakemake.params.response_prefix + celltype.lower(),
            },
        ],
    }

    if snakemake.params.pre_prompt_topgenes:  # type: ignore [reportUndefinedVariable]
        pre_prompt = copy.deepcopy(snakemake.params.pre_prompt_topgenes)
        pre_prompt[0]["value"] = pre_prompt[0]["value"].format(
            ", ".join(top_genes.loc[cell_id])
        )
        obj["conversations"] = pre_prompt + obj["conversations"]  # type: ignore [reportUndefinedVariable]
    else:
        obj["conversations"][0]["value"] += "\n<image>"
        obj["image"] = cell_id

    conversations.append(obj)

# Save the conversations to a JSON file
with open(snakemake.output.conversation_dataset, "w") as f:
    json.dump(conversations, f, indent=2)
