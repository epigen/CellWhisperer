#!/usr/bin/env python
"""
Compute per-core base CellWhisperer model metrics: retrieval + contrastive loss.

Loads the frozen CellWhisperer model, runs inference on a TMA dataset to extract
image and transcriptome embeddings, then computes per-core:
  - Image<->Transcriptome retrieval metrics (recall@k, precision, accuracy, F1, AUROC)
  - Contrastive loss (per-sample, then aggregated per core)

Usage:
    python compute_core_base_metrics.py \
        --checkpoint /path/to/v1.ckpt \
        --dataset_name lymphoma_cosmx_large_TMA2 \
        --batch_size 64 \
        --output_retrieval core_retrieval.csv \
        --output_loss core_loss.csv
"""

import pyarrow  # needed first on some systems
import argparse
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from cellwhisperer.jointemb.cellwhisperer_lightning import TranscriptomeTextDualEncoderLightning
from cellwhisperer.jointemb.dataset import JointEmbedDataModule
from cellwhisperer.utils.inference import score_left_vs_right
from cellwhisperer.validation.zero_shot.functions import prepare_metrics_and_labels


def get_core_ids_for_orig_ids(predictions_path, orig_ids):
    """Map orig_ids -> core_id using the predictions h5ad (which has core_id in obs)."""
    print(f"Loading predictions h5ad for core_id mapping: {predictions_path}")
    adata = anndata.read_h5ad(predictions_path, backed="r")

    # Build a lookup from obs index -> core_id
    core_id_map = dict(zip(adata.obs.index, adata.obs["core_id"]))
    core_ids = np.array([core_id_map.get(oid, "") for oid in orig_ids], dtype=str)
    n_mapped = np.sum(core_ids != "")
    print(f"  Mapped {n_mapped}/{len(orig_ids)} orig_ids to core_ids")
    return core_ids


def compute_per_sample_contrastive_loss(image_embeds, transcriptome_embeds, temperature):
    """
    Compute per-sample symmetric contrastive loss.
    Returns array of shape (n_samples,) with the average of row and column loss.
    """
    n = image_embeds.shape[0]
    logits = torch.matmul(image_embeds, transcriptome_embeds.T) * temperature
    targets = torch.arange(n, device=logits.device)

    # Row loss: for each image, find its transcriptome
    row_loss = F.cross_entropy(logits, targets, reduction="none")
    # Column loss: for each transcriptome, find its image
    col_loss = F.cross_entropy(logits.T, targets, reduction="none")

    return ((row_loss + col_loss) / 2.0).cpu().numpy()


def compute_retrieval_for_core(image_embeds, transcriptome_embeds, logit_scale):
    """Compute retrieval metrics for a set of embeddings from one core."""
    n = image_embeds.shape[0]
    correct_indices = list(range(n))

    # Image -> Transcriptome
    scores_i2t, _ = score_left_vs_right(
        left_input=image_embeds,
        right_input=transcriptome_embeds,
        logit_scale=logit_scale,
        model=None,
        average_mode=None,
        grouping_keys=None,
        batch_size=min(128, n),
        score_norm_method=None,
        use_image_data=False,
    )

    # Transcriptome -> Image
    scores_t2i, _ = score_left_vs_right(
        left_input=transcriptome_embeds,
        right_input=image_embeds,
        logit_scale=logit_scale,
        model=None,
        average_mode=None,
        grouping_keys=None,
        batch_size=min(128, n),
        score_norm_method=None,
        use_image_data=False,
    )

    metrics_i2t, _ = prepare_metrics_and_labels(
        scores=scores_i2t,
        left_input=image_embeds,
        right_input=transcriptome_embeds,
        correct_right_idx_per_left=correct_indices,
        average_mode=None,
        grouping_keys=None,
        right_as_classes=False,
        report_per_class_metrics=False,
    )

    metrics_t2i, _ = prepare_metrics_and_labels(
        scores=scores_t2i,
        left_input=transcriptome_embeds,
        right_input=image_embeds,
        correct_right_idx_per_left=correct_indices,
        average_mode=None,
        grouping_keys=None,
        right_as_classes=False,
        report_per_class_metrics=False,
    )

    result = {}
    for metric_name, val in metrics_i2t.items():
        result[f"i2t_{metric_name}"] = val.item() if hasattr(val, "item") else float(val)
    for metric_name, val in metrics_t2i.items():
        result[f"t2i_{metric_name}"] = val.item() if hasattr(val, "item") else float(val)

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--predictions", type=Path, required=True,
                        help="Predictions h5ad file (used for core_id mapping)")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--nproc", type=int, default=8)
    parser.add_argument("--min_cells", type=int, default=50)
    parser.add_argument("--output_retrieval", type=Path, required=True)
    parser.add_argument("--output_loss", type=Path, required=True)
    args = parser.parse_args()

    args.output_retrieval.parent.mkdir(parents=True, exist_ok=True)
    args.output_loss.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    print(f"Loading CellWhisperer checkpoint: {args.checkpoint}")
    model = TranscriptomeTextDualEncoderLightning.load_from_checkpoint(str(args.checkpoint))
    model.eval()
    model.to(device)

    temperature = model.model.discriminator.temperature.exp().detach().cpu()
    print(f"Temperature: {temperature.item():.4f}")

    # Set up data module
    print(f"Setting up data module for {args.dataset_name}")
    dm = JointEmbedDataModule(
        tokenizer="bert",
        transcriptome_processor="mlp",
        image_processor="uni2",
        dataset_names=args.dataset_name,
        batch_size=args.batch_size,
        nproc=args.nproc,
        train_fraction=1.0,  # use all data (no train/val split)
        use_disk_loading=True,
    )
    dm.setup("predict")
    dataloader = dm.predict_dataloader()
    print(f"Dataloader ready: {len(dataloader)} batches")

    # Get orig_ids directly from the dataset (not from batches, since disk-loaded
    # samples don't include orig_ids in the returned dict)
    dataset = dataloader.dataset
    if hasattr(dataset, "orig_ids"):
        all_orig_ids = list(dataset.orig_ids)
    elif hasattr(dataset, "datasets"):
        # ConcatDataset
        all_orig_ids = []
        for ds in dataset.datasets:
            all_orig_ids.extend(ds.orig_ids)
    else:
        raise RuntimeError("Cannot extract orig_ids from dataset")
    print(f"Got {len(all_orig_ids)} orig_ids from dataset")

    # Collect embeddings
    all_image_embeds = []
    all_transcriptome_embeds = []

    print("Extracting embeddings...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader)):
            # Move batch to device
            batch_device = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch_device[k] = v.to(device)
                else:
                    batch_device[k] = v

            output = model(**batch_device)

            if output.image_embeds is not None:
                all_image_embeds.append(output.image_embeds.cpu())
            if output.transcriptome_embeds is not None:
                all_transcriptome_embeds.append(output.transcriptome_embeds.cpu())

    image_embeds = torch.cat(all_image_embeds, dim=0)
    transcriptome_embeds = torch.cat(all_transcriptome_embeds, dim=0)
    print(f"Collected embeddings: {image_embeds.shape[0]} samples, dim={image_embeds.shape[1]}")
    assert len(all_orig_ids) == image_embeds.shape[0], \
        f"orig_ids count ({len(all_orig_ids)}) != embeddings count ({image_embeds.shape[0]})"

    # Map orig_ids -> core_id using predictions h5ad
    core_ids = get_core_ids_for_orig_ids(args.predictions, all_orig_ids)

    # Filter out unmapped samples
    mapped_mask = np.array([cid != "" for cid in core_ids])
    if not mapped_mask.all():
        print(f"Dropping {(~mapped_mask).sum()} unmapped samples")
        image_embeds = image_embeds[mapped_mask]
        transcriptome_embeds = transcriptome_embeds[mapped_mask]
        core_ids = core_ids[mapped_mask]

    unique_cores = np.unique(core_ids)
    tma = args.dataset_name.rsplit("_", 1)[1]
    print(f"Found {len(unique_cores)} unique cores in {tma}")

    # Compute per-core metrics
    retrieval_rows = []
    loss_rows = []

    for core in unique_cores:
        mask = core_ids == core
        n_cells = mask.sum()

        if n_cells < args.min_cells:
            print(f"  Skipping core {core} ({n_cells} cells < {args.min_cells})")
            continue

        core_image = image_embeds[mask]
        core_transcriptome = transcriptome_embeds[mask]

        # --- Retrieval ---
        print(f"  Core {core}: {n_cells} cells, computing retrieval...")
        retrieval_metrics = compute_retrieval_for_core(
            core_image, core_transcriptome, temperature
        )
        retrieval_row = {"tma": tma, "core_id": core, "n_cells": int(n_cells)}
        retrieval_row.update(retrieval_metrics)
        retrieval_rows.append(retrieval_row)

        # --- Contrastive Loss ---
        per_sample_loss = compute_per_sample_contrastive_loss(
            core_image, core_transcriptome, temperature
        )
        loss_rows.append({
            "tma": tma,
            "core_id": core,
            "n_cells": int(n_cells),
            "mean_loss": np.mean(per_sample_loss),
            "median_loss": np.median(per_sample_loss),
            "std_loss": np.std(per_sample_loss),
        })

    # Save
    df_retrieval = pd.DataFrame(retrieval_rows)
    df_retrieval.to_csv(args.output_retrieval, index=False)
    print(f"Saved retrieval metrics for {len(df_retrieval)} cores to {args.output_retrieval}")

    df_loss = pd.DataFrame(loss_rows)
    df_loss.to_csv(args.output_loss, index=False)
    print(f"Saved loss metrics for {len(df_loss)} cores to {args.output_loss}")


if __name__ == "__main__":
    main()
