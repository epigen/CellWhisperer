import logging
import numpy as np
import scib
from anndata import AnnData
from typing import Dict
import numpy as np
import scanpy as sc


# FROM: https://github.com/microsoft/zero-shot-scfoundation/blob/f0353a49cb6aff5c8105cc6ddd755efa8fbab98d/sc_foundation_evals/utils.py#L16
# MODIFIED wrapper for all scib metrics from 
# https://github.com/bowang-lab/scGPT/blob/5a69912232e214cda1998f78e5b4a7b5ef09fe06/scgpt/utils/util.py#L267
def eval_scib_metrics(
    adata: AnnData,
    batch_key: str = "str_batch",
    label_key: str = "cell_type",
    embedding_key: str = "X_scGPT"
) -> Dict:
    
    # if adata.uns["neighbors"] exists, remove it to make sure the optimal 
    # clustering is calculated for the correct embedding
    # print a warning for the user
    if "neighbors" in adata.uns:        
        # logging.warning(f"neighbors in adata.uns found \n {adata.uns['neighbors']} "
        #             "\nto make sure the optimal clustering is calculated for the "
        #             "correct embedding, removing neighbors from adata.uns."
        #             "\nOverwriting calculation of neighbors with "
        #             f"sc.pp.neighbors(adata, use_rep={embedding_key}).")
        adata.uns.pop("neighbors", None)
        sc.pp.neighbors(adata, use_rep=embedding_key)
        # logging.info("neighbors in adata.uns removed, new neighbors calculated: "
        #          f"{adata.uns['neighbors']}")


    # in case just one batch scib.metrics.metrics doesn't work 
    # call them separately
    results_dict = dict()

    res_max, nmi_max, nmi_all = scib.metrics.clustering.opt_louvain(
            adata,
            label_key=label_key,
            cluster_key="cluster",
            use_rep=embedding_key,
            function=scib.metrics.nmi,
            plot=False,
            verbose=False,
            inplace=True,
            force=True,
    )

    # was results_dict["NMI_cluster/label"], I changed the name to improve wandb integration
    results_dict["NMI_cluster__label"] = scib.metrics.nmi(
        adata, 
        "cluster",
        label_key,
        "arithmetic",
        nmi_dir=None
    )

    # was results_dict["ARI_cluster/label"], I changed the name to improve wandb integration
    results_dict["ARI_cluster__label"] = scib.metrics.ari(
        adata, 
        "cluster", 
        label_key
    )

    results_dict["ASW_label"] = scib.metrics.silhouette(
        adata, 
        label_key, 
        embedding_key, 
        "euclidean"
    )   

    results_dict["graph_conn"] = scib.metrics.graph_connectivity(
        adata,
        label_key=label_key
    )
    

    # Calculate this only if there are multiple batches
    if len(adata.obs[batch_key].unique()) > 1:

        # PP: This seems to be not really useful, e.g. scib.metrics.metrics does not even compute this
        # results_dict["ASW_batch"] = scib.metrics.silhouette(
        #     adata,
        #     batch_key,
        #     embedding_key,
        #     "euclidean"
        # )

        # PP: This is what scib.metrics.metrics reports as "A
        # SW_label/batch" and internally it calls it "asw_batch
        # I think this is the better way to calculate it.
        # was results_dict["ASW_label/batch"], I changed the name to improve wandb integration
        results_dict["ASW_label__batch"] = scib.metrics.silhouette_batch(
            adata, 
            batch_key,
            label_key, 
            embed=embedding_key, 
            metric="euclidean",
            return_all=False,
            verbose=False
        )

        results_dict["PCR_batch"] = scib.metrics.pcr(
            adata,
            covariate=batch_key,
            embed=embedding_key,
            recompute_pca=True,
            n_comps=50,
            verbose=False
        )

    results_dict["avg_bio"] = np.mean(
        [
            results_dict["NMI_cluster__label"],
            results_dict["ARI_cluster__label"],
            results_dict["ASW_label"],
        ]
    )

    logging.debug(
        "\n".join([f"{k}: {v:.4f}" for k, v in results_dict.items()])
    )

    # remove nan value in result_dict
    results_dict = {k: v for k, v in results_dict.items() if not np.isnan(v)}

    # remove "neighbor" from adata.uns
    if "neighbors" in adata.uns:
        adata.uns.pop("neighbors", None)

    return results_dict
