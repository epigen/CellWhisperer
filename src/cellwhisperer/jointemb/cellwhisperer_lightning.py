"""
Wraps lightning and transformers logic, providing bare configs to the user/CLI
"""

import warnings

from lightning import LightningModule
from cellwhisperer.validation import initialize_validation_functions
from cellwhisperer.config import model_path_from_name
from cellwhisperer.jointemb.loss.config import LossConfig
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim import AdamW
from cellwhisperer.jointemb.model import (
    TranscriptomeTextDualEncoderConfig,
    TranscriptomeTextDualEncoderModel,
    CLIPOutput,
)
from cellwhisperer.jointemb.processing import TranscriptomeTextDualEncoderProcessor

from lightning.pytorch.cli import OptimizerCallable, LRSchedulerCallable
from wandb import Artifact, Table

from typing import Optional, Union, Dict, List
from pathlib import Path
import torch
import copy
from functools import partial
import logging

from cellwhisperer.jointemb.regularization import InputRegularization

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r".*is deprecated and will be removed in a future version.*",
)


class TranscriptomeTextDualEncoderLightning(LightningModule):
    def __init__(
        self,
        model_config: Dict,
        loss_config: Dict,
        val_batch_size: int = 32,
        gauss_noise_std: float = 0.0,
        max_epochs: int = 100,
        optimizer: OptimizerCallable = torch.optim.AdamW,
        scheduler: LRSchedulerCallable = lambda optimizer: CosineAnnealingLR(
            optimizer, T_max=100, eta_min=0
        ),  # TODO need to find a solution to make T_max flexible here
        learning_rate: float = 1e-3,
        lr_warmup_steps: int = 100,
    ):
        """
        Args:
            model_config: configuration for the model. Can be a dict or a TranscriptomeTextDualEncoderConfig object.
            loss_config: configuration for the loss. Can be a dict or a LossConfig object.
            val_batch_size: batch size to used for dedicated validation functions. 32 should be feasible on most GPUs.
            gauss_noise_std: Currently inactive: standard deviation of the gaussian noise to add to the input embeddings (features). if 0.0 or None noise won't be added to training embeddings
            max_epochs: maximum number of epochs to train for
            optimizer: optimizer to use. Must be a callable that returns an optimizer object.
            scheduler: scheduler to use. Must be a callable that returns a scheduler object.
            learning_rate: learning rate to use for the optimizer
        """
        super(TranscriptomeTextDualEncoderLightning, self).__init__()

        # Convert dicts to the respective object types (checkpoints contain the Config object, but training passes dicts)
        if not isinstance(model_config, TranscriptomeTextDualEncoderConfig):
            model_config = TranscriptomeTextDualEncoderConfig(**model_config)

        if not isinstance(loss_config, LossConfig):
            loss_config = LossConfig(**loss_config)

        self.model = TranscriptomeTextDualEncoderModel(
            config=model_config,
        )
        self.max_epochs = max_epochs
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.learning_rate = learning_rate
        self.lr_warmup_steps = lr_warmup_steps

        self.loss_config = loss_config
        self.loss_functions = self.loss_config.configure_losses(
            self.model.discriminator
        )

        self.input_regularization = InputRegularization(gauss_noise_std=gauss_noise_std)
        self.val_batch_size = val_batch_size

        self.save_hyperparameters()

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: str,
        **kwargs,
    ):
        """
        Because our checkpoints might not contain the foundation models, we need to load them separately and then load the checkpoint.

        NOTE: we only need to do this explicit sub-model loading, if one of the models is frozen (which is usually the case)
        """
        model = super().load_from_checkpoint(checkpoint_path, **kwargs)
        # Park state_dict
        state_dict = model.state_dict().copy()

        # if transcriptome_model_directory is None:
        transcriptome_model_directory = model_path_from_name(
            model.model.transcriptome_model.config.model_type
        )
        # if text_model_name_or_path is None:
        text_model_name_or_path = model_path_from_name(
            model.model.text_model.config.model_type
        )

        # make sure that pretrained models are loaded
        model.load_pretrained_models(
            transcriptome_model_directory=transcriptome_model_directory,
            text_model_name_or_path=text_model_name_or_path,
        )

        # Restore state_dict
        model.load_state_dict(state_dict)

        return model

    def load_pretrained_models(
        self,
        transcriptome_model_directory: Optional[str],
        text_model_name_or_path: Optional[str],
    ):
        """
        This method exhibits an interface to load the pretrained models after initialization. This allows loading of pretrained weights from a checkpoint, without initializing these models..
        """

        kwargs = self.model.config.to_dict()
        if transcriptome_model_directory is None:
            kwargs["transcriptome_model"] = self.model.transcriptome_model

        if text_model_name_or_path is None:
            kwargs["text_model"] = self.model.text_model

        self.model = (
            TranscriptomeTextDualEncoderModel.from_transcriptome_text_pretrained(
                transcriptome_model_name_or_path=transcriptome_model_directory,
                text_model_name_or_path=text_model_name_or_path,
                **kwargs,
            )
        )
        self.loss_functions = self.loss_config.configure_losses(
            self.model.discriminator
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor],
        expression_tokens: Optional[torch.FloatTensor] = None,
        expression_token_lengths: Optional[torch.LongTensor] = None,
        expression_gene: Optional[torch.LongTensor] = None,
        expression_expr: Optional[torch.LongTensor] = None,
        expression_key_padding_mask: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,  # token_type_ids
    ) -> CLIPOutput:
        return self.model(
            input_ids=input_ids,
            expression_tokens=expression_tokens,
            expression_token_lengths=expression_token_lengths,
            expression_gene=expression_gene,
            expression_expr=expression_expr,
            expression_key_padding_mask=expression_key_padding_mask,
            attention_mask=attention_mask,
            return_dict=True,
            **kwargs,
        )

    def process_step(self, batch, batch_idx, step_type):
        outputs = self(**batch)

        # outputs = {k: v.to(device) if v is not None else None for k, v in outputs.items()}
        combined_loss = torch.tensor(0.0, device=self.device)

        for loss in self.loss_functions:
            # Calculate the loss for the current batch using the specific loss function.
            loss_value = loss["fn"](
                **outputs,
                transcriptome_weights=batch.get("transcriptome_weights"),
                annotation_weights=batch.get("annotation_weights"),
            )
            combined_loss = combined_loss + (loss_value * loss["lambda"])

            # Log the individual loss value for monitoring.
            self.log(
                f"{step_type}/{loss['name']}_loss",
                loss_value,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True,  # NOTE: might lead to more synchronization overhead
            )

        # After processing all loss functions, log the total combined loss.
        self.log(
            f"{step_type}/loss",
            combined_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,  # NOTE: might lead to more synchronization overhead
        )
        # NOTE: this is a hack to avoid NaNs in the loss. We should fix this properly.
        if torch.isnan(combined_loss):
            logging.warning("NaN loss detected. Setting loss to 0.0.")
            return torch.tensor(0.0, device=self.device)

        return combined_loss

    def training_step(self, batch, batch_idx):
        return self.process_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self.process_step(batch, batch_idx, "val")

    def setup(self, stage=None):
        # We do this here because during __init__ self.trainer is not yet available
        self.validation_functions = initialize_validation_functions(
            batch_size=self.val_batch_size,
            transcriptome_model_type=self.model.transcriptome_model.config.model_type,
            text_model_type=self.model.text_model.config.model_type,
            val_dataloader=self.trainer.datamodule.val_dataloader(),
        )

    def on_validation_epoch_end(self):
        # For convenience (speed), I disable this when "fast_dev_run" is enabled
        if not self.trainer.fast_dev_run:
            logging.info("Running validation functions")
            for val_fn_name, val_fn in self.validation_functions.items():
                with torch.no_grad():  # necessary, despite model being in eval mode
                    val_results = val_fn(self.model)
                    val_metrics = val_results[0]
                for metric_name, metric_value in val_metrics.items():
                    # NOTE enabling sync_dist requires the logged metric to be on GPU
                    # In our case it doesn't matter, because we are training on a single GPU/node at the moment
                    # see https://github.com/Lightning-AI/pytorch-lightning/issues/18803
                    self.log(
                        f"valfn_{val_fn_name}/{metric_name}",
                        metric_value,
                        on_step=False,
                        on_epoch=True,
                        prog_bar=False,
                        logger=True,
                        # sync_dist=True,
                    )
                    if val_results[1] is not None:
                        artifact = Artifact(
                            f"valfn_{val_fn_name}_per_celltype_run{self.logger.experiment.id}",
                            type="performance_metrics",
                        )
                        # turn the df val_results[1] into a wandb table:
                        table = Table(dataframe=val_results[1].reset_index())
                        artifact.add(table, "performance_metrics")
                        self.logger.experiment.log_artifact(artifact)

    def on_train_epoch_end(self):
        """
        This function is called at the end of each epoch, after training AND validation.

        Store the cache of the model, after the first epoch to make it available to other runs as early as possible.
        """

        if self.trainer.current_epoch == 0:
            logging.debug("Storing cache of model")
            self.model.store_cache()

    def on_train_end(self):
        """
        Store the cache of the model, at the end of training. Since we drop the last batch (in training), we don't get all samples in the first batch
        """
        logging.debug("Storing cache of model")
        self.model.store_cache()

    def test_step(self, batch, batch_idx):
        return self.process_step(batch, batch_idx, "test")

    def configure_optimizers(self):
        """
        Use AdamW optimizer for decoupled weight decay. Apply decay weights only (e.g. not biases), as indicated by the CLIP paper(s).

        Also implement cosine learning rate schedule, as indicated by the CLIP paper(s).

        https://arxiv.org/pdf/2303.15343.pdf (Fig. 4) indicates better performance when NOT decaying the weights of the 'U'nlocked pretrained model. Other CLIP papers indicated that only the weigths not the bias should be decayed.

        Alternative could be to use AdaBelief (which works well according to "ItalianClip")
        """

        # Decay weights as indicated in docstring
        # NOTE subject this to ablation study
        decay_selector = lambda name: "weight" in name and (
            (
                self.model.config.locking_mode[0] == "u"
                and ".transcriptome_model." in name
            )
            or (self.model.config.locking_mode[1] == "u" and ".text_model." in name)
        )

        weight_params = [
            param for name, param in self.named_parameters() if decay_selector(name)
        ]
        other_params = [
            param for name, param in self.named_parameters() if not decay_selector(name)
        ]

        # Workaround to all using --config <file> (otherwise it does not work :())
        logging.warning("passed optimizer arg is being ignored in favor of adamw")
        optimizer = AdamW(
            [
                {"params": weight_params, "weight_decay": 0.01},
                {"params": other_params, "weight_decay": 0.0},
            ],
            lr=self.learning_rate,
            betas=(0.9, 0.98),  # SigLiT recommends even 0.95 for beta2
        )
        scheduler = {
            "scheduler": self.scheduler(optimizer),
            "interval": "epoch",
            "monitor": "val_loss",
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        """
        Override `optimizer_step` to allow learning-rate warmup
        """
        if self.trainer.global_step < self.lr_warmup_steps:
            lr_scale = min(
                1.0, float(self.trainer.global_step + 1) / self.lr_warmup_steps
            )
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.learning_rate

        for pg in optimizer.param_groups:
            self.log(
                "train/learning_rate",
                pg["lr"],
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
            break  # (only the first one is needed)

        super(TranscriptomeTextDualEncoderLightning, self).optimizer_step(
            epoch, batch_idx, optimizer, optimizer_closure
        )

    # inference API
    def embed_texts(self, texts: List[str], chunk_size=64):
        """
        Embed the given texts into the LLM space
        """

        processor = TranscriptomeTextDualEncoderProcessor(
            self.model.transcriptome_model.config.model_type,
            model_path_from_name(self.model.text_model.config.model_type),
        )

        tokenizer = processor.tokenizer

        ret_list = []
        for chunk in [
            texts[i : i + chunk_size] for i in range(0, len(texts), chunk_size)
        ]:
            text_tokens = tokenizer(chunk, return_tensors="pt", padding=True)
            for k, v in text_tokens.items():
                text_tokens[k] = v.to(self.model.device)

            # Compute text embeddings
            _, text_embeds = self.model.get_text_features(**text_tokens)
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
            ret_list.append(text_embeds)

        return torch.cat(ret_list, dim=0)
