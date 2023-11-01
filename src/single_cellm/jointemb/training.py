#!/usr/bin/env python
# coding: utf-8

import torch
import argparse
import logging
from single_cellm.jointemb.model import TranscriptomeTextDualEncoderConfig
from single_cellm.jointemb.lightning import TranscriptomeTextDualEncoderLightning
from pathlib import Path
import subprocess
import yaml
from pathlib import Path
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from single_cellm.jointemb.dataset import JointEmbedDataModule
import os
import anndata

torch.set_float32_matmul_precision("high")  # speed up on ampere-level GPUs
torch.multiprocessing.set_sharing_strategy("file_system")
# warnings.filterwarnings("ignore")  <- only enable with a proper comment on why this is necessary


PROJECT_DIR = Path(
    subprocess.check_output(["git", "rev-parse", "--show-toplevel"], cwd=os.getcwd())
    .decode("utf-8")
    .strip()
)

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="daniel", type=str, help="Dataset name")
parser.add_argument(
    "--max_epochs", default=300, type=int, help="Maximum number of epochs"
)
parser.add_argument(
    "--run_name",
    default="jointEmbed_15k_unfrozen_shuffledval",
    type=str,
    help="Run name",
)
parser.add_argument(
    "--freeze_pretrained",
    default=False,
    type=bool,
    help="Freeze pretrained model or not",
)

args = parser.parse_args()

with open(PROJECT_DIR / "config.yaml") as f:
    config = yaml.safe_load(f)
MODEL_DIR = PROJECT_DIR / config["paths"]["jointemb_models"]
MODEL_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR = PROJECT_DIR / config["paths"]["wandb_logs"]
LOG_DIR.mkdir(parents=True, exist_ok=True)

config_transcriptome = {"model_type": "geneformer"}
config_text = {"model_type": "biogpt"}

model_config = TranscriptomeTextDualEncoderConfig(
    projection_dim=512,
    logit_scale_init_value=2.6592,
    transcriptome_config=config_transcriptome,
    text_config=config_text,
)
device = torch.device("cuda")

pl_model = TranscriptomeTextDualEncoderLightning(config=model_config)
try:
    pl_model.load_pretrained_models(PROJECT_DIR / "resources" / "geneformer-12L-30M")
except FileNotFoundError:
    logging.error(
        "Unable to fine geneformer model. Please download first (see `rna` snakemake pipeline)"
    )
    raise
if args.freeze_pretrained:
    pl_model.freeze_pretrained()  # optional
pl_model.to(device)

dm = JointEmbedDataModule(args.dataset, batch_size=32)
dm.setup(stage="fit")
dm.prepare_data()

logger = WandbLogger(
    save_dir=LOG_DIR, name=args.run_name, project="JointEmbed_Training"
)
early_stop = EarlyStopping(
    monitor="val_loss", min_delta=1e-5, patience=20, verbose=False, mode="min"
)
checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    mode="min",
    save_top_k=2,
    save_last=True,
    filename="{epoch}-{val_loss:.2f}",
)

# testing, run only few samples
trainer = Trainer(
    max_epochs=args.max_epochs,
    # max_epochs=1,
    # limit_train_batches=5,
    # limit_val_batches=5,
    accelerator="gpu",
    default_root_dir=LOG_DIR,  # TODO do we really need this param?
    logger=logger,
    enable_progress_bar=True,
    callbacks=[checkpoint_callback, early_stop],
)
trainer.fit(pl_model, datamodule=dm)

print(f"Best model saved at {checkpoint_callback.best_model_path}")
trainer.save_checkpoint(MODEL_DIR / f"{args.run_name}_final.ckpt")
