"""
Deduplication of datasets through clustering on the level (BERT) language model embeddings


NOTE: Currently we're using the subsetted adatas. We could have used full adatas and simple add a column that indicates the cluster center. (This is also already prepared for in the deduplicate_dataset function. The obs column is always called "is_in_{deduplicated_dataset_name}")
"""
from cellwhisperer.config import get_path, config
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from scipy.cluster import hierarchy
from collections import defaultdict
from scipy.spatial.distance import cosine
from collections import Counter
import umap
import matplotlib.pyplot as plt
import anndata
import os
from typing import Dict
import lightning as pl

pl.seed_everything(42)


os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["OMP_NUM_THREADS"] = "10"


def _cluster(
    embeddings,
    annotations_text,
    annotations_sample_ids,
    n_clusters,
    outdir,
    save_jsons=False,
    verbose=False,
):
    # Cluster the embeddings
    Z = hierarchy.linkage(embeddings, "average", metric="cosine")
    cluster_assignment = hierarchy.fcluster(Z, n_clusters, criterion="maxclust")
    print(f"Number of clusters: {len(set(cluster_assignment))}")

    # get the cluster assignments
    clustered_sentences = defaultdict(list)
    for sentence_id, cluster_id in enumerate(cluster_assignment):
        clustered_sentences[cluster_id].append(annotations_text[sentence_id])

    # find the center-most sentence for each cluster
    cluster_center_text_list = []
    cluster_center_id_list = []
    cluster_center_sample_id_list = []

    for cluster_id in clustered_sentences.keys():
        if cluster_id == -1:
            continue
        cluster_embeddings = embeddings[cluster_assignment == cluster_id]
        global_indices_this_cluster = np.where(cluster_assignment == cluster_id)
        cluster_center_embedding = np.mean(cluster_embeddings, axis=0)

        # renormalize the cluster center embedding
        cluster_center_embedding = cluster_center_embedding / np.linalg.norm(
            cluster_center_embedding
        )

        min_distance = 1
        global_index_this_cluster_center = None
        for i, sentence_embedding in enumerate(cluster_embeddings):
            distance = cosine(cluster_center_embedding, sentence_embedding)
            if distance < min_distance:
                min_distance = distance
                global_index_this_cluster_center = global_indices_this_cluster[0][i]
        cluster_center_id_list.append(global_index_this_cluster_center)
        cluster_center_text_list.append(
            annotations_text[global_index_this_cluster_center]
        )
        cluster_center_sample_id_list.append(
            annotations_sample_ids[global_index_this_cluster_center]
        )

    if verbose:
        # get the number of members per cluster
        for cluster_id, count in Counter(cluster_assignment).most_common():
            print(f"Cluster {cluster_id}: {count} members")

        # Some examples of clusters
        for cluster_id in range(1, 10):
            print("\n\n")
            print(f"Cluster {cluster_id}:")
            for i, sentence_id in enumerate(
                np.where(cluster_assignment == cluster_id)[0]
            ):
                print(f"{i+1}. {annotations_text[sentence_id]}")
            print("\n\n")

        # visualize the data with UMAP
        reducer = umap.UMAP()
        umap_embedding = reducer.fit_transform(embeddings)

        # color by the cluster labels
        plt.scatter(
            umap_embedding[:, 0],
            umap_embedding[:, 1],
            c=cluster_assignment,
            cmap="Spectral",
            s=0.01,
        )
        plt.gca().set_aspect("equal", "datalim")
        # write the cluster ID next to the point for the center-most sentence in each cluster
        for i, id in enumerate(cluster_center_id_list):
            x = umap_embedding[id, 0]
            y = umap_embedding[id, 1]
            plt.annotate(str(i), xy=(x, y), fontsize=7)
        plt.show()

        # print the cluster center text
        for i, text in enumerate(cluster_center_text_list):
            print(f"{i}. {text}")

    # save: the cluster assignments, and the cluster center ids /  the cluster center text as dict
    if save_jsons:
        with open(f"{outdir}/cluster_assignments.json", "w") as f:
            json.dump(cluster_assignment.tolist(), f)
        with open(f"{outdir}/cluster_centers.cluster_ids_and_text.json", "w") as f:
            json.dump(
                {
                    str(id): text
                    for id, text in zip(
                        cluster_center_id_list, cluster_center_text_list
                    )
                },
                f,
            )
        with open(f"{outdir}/processed_annotations.json", "w") as f:
            json.dump(
                {
                    sample_id: text
                    for sample_id, text in zip(
                        cluster_center_sample_id_list, cluster_center_text_list
                    )
                },
                f,
            )
        with open(f"{outdir}/clustered_sentences.json", "w") as f:
            json.dump({str(k): v for k, v in clustered_sentences.items()}, f)

    return cluster_center_sample_id_list, cluster_assignment


def deduplicate_dataset(
    dataset: str = "human_disease",
    annotations: Dict[str, str] = None,
    n_clusters_list: list = [100, 300, 1000],
    model_name: str = "dmis-lab/biobert-v1.1",
    pooling_mode_cls_token: bool = False,
) -> None:
    """
    Deduplicate the dataset by clustering the annotations using a sentence transformer model. Saves, for three stringency levels: \
    - the cluster assignments \
    - the cluster center ids and text \
    - the cluster center sample ids and text \
    - the clustered sentences \
    - the deduplicated dataset (anndata)

    Parameters
    ----------
    dataset : Name of the dataset to deduplicate. If annotations is None, the annotations are loaded from the anndata file for this dataset.
    annotations : Dict[str, str]. A dictionary of sample ids and text annotations. If None, the annotations are loaded from the anndata file for this dataset.
    n_clusters_list : list. The n_clusters to use for clustering. Strict, normal and loose clustering will be performed based on those
    model_name : str. The name of the transformer model to use
    pooling_mode_cls_token : bool. Whether to use the CLS token for pooling or the mean of all tokens

    """

    print(
        f"Running on dataset {dataset} with model {model_name} and pooling_mode_cls_token {pooling_mode_cls_token}"
    )

    # Load the anndata:
    adata = anndata.read_h5ad(get_path(["paths", "full_dataset"], dataset=dataset))

    if annotations is None:
        annotations_text = adata.obs[config["anndata_label_name"]].values.to_list()
        annotations_sample_ids = adata.obs.index.to_list()
    else:
        annotations_text = list(annotations.values())
        annotations_sample_ids = list(annotations.keys())

    # Load model
    from sentence_transformers import SentenceTransformer, models

    word_embedding_model = models.Transformer(model_name)

    pooling_model = models.Pooling(
        word_embedding_model.get_word_embedding_dimension(),
        pooling_mode_cls_token=pooling_mode_cls_token,
        pooling_mode_mean_tokens=not pooling_mode_cls_token,
    )
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    model.eval()

    # Get embeddings
    embeddings = model.encode(
        annotations_text, show_progress_bar=True, convert_to_numpy=True
    )

    # Normalize embeddings
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    for n_clusters, name in zip(n_clusters_list, ["strictly", "normally", "loosely"]):
        dedup_dataset_name = f"{dataset}_{name}_deduplicated_{model_name.replace('/', '_')}_{'CLS_pooling' if pooling_mode_cls_token else 'mean_pooling'}"
        outdir = get_path(["paths", "full_dataset"], dataset=dedup_dataset_name).parent
        os.makedirs(outdir, exist_ok=True)
        sample_ids_to_keep, cluster_assignment = _cluster(
            embeddings,
            annotations_text,
            annotations_sample_ids,
            n_clusters,
            outdir,
            save_jsons=True,
            verbose=True,
        )
        adata.obs[f"cluster_assignment_{dedup_dataset_name}"] = cluster_assignment
        adata.obs[f"is_in_{dedup_dataset_name}"] = [
            x in sample_ids_to_keep for x in adata.obs.index
        ]

        adata_subset = adata[adata.obs.index.isin(sample_ids_to_keep)]
        adata_subset.write_h5ad(
            get_path(["paths", f"full_dataset"], dataset=dedup_dataset_name)
        )
    adata.write_h5ad(get_path(["paths", f"full_dataset"], dataset=dataset))


if __name__ == "__main__":
    evaluate_different_settings = False

    if evaluate_different_settings:
        for model in [
            "bert-base-uncased",
            "gsarti/biobert-nli",
            "dmis-lab/biobert-v1.1",
            "gsarti/biobert-nli",
            "pritamdeka/S-BioBert-snli-multinli-stsb",
        ]:
            for pooling_mode_cls_token in [False, True]:
                deduplicate_dataset(
                    dataset="human_disease",
                    model_name=model,
                    pooling_mode_cls_token=pooling_mode_cls_token,
                )
    else:
        deduplicate_dataset(dataset="human_disease", pooling_mode_cls_token=True)

    # deduplicate the immgen dataset
    adata = anndata.read_h5ad(get_path(["paths", "full_dataset"], dataset="immgen"))
    adata.obs["replicate"] = [x[1] for x in adata.obs.index.str.split("#")]
    keep_idx = [i for i, x in enumerate(adata.obs["replicate"]) if x == "1"]
    adata_subset = adata[keep_idx]
    outpath = get_path(["paths", f"full_dataset"], dataset="immgen_deduplicated")
    os.makedirs(outpath.parent, exist_ok=True)
    adata_subset.write_h5ad(outpath)

    # adata.obs["is_in_immgen_deduplicated"] = [x in keep_idx for x in range(len(adata))]
    # adata.write_h5ad(get_path(["paths", f"full_dataset"], dataset="immgen"))
