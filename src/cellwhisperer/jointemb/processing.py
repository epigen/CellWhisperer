# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Processor class for TranscriptomeTextDualEncoder
"""

import warnings
import logging
from typing import Any, Union

logger = logging.getLogger(__name__)

from transformers import AutoTokenizer
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import BatchEncoding
from .geneformer_model import GeneformerTranscriptomeProcessor
from .scgpt_model import ScGPTTranscriptomeProcessor


class TranscriptomeTextDualEncoderProcessor(ProcessorMixin):
    r"""
    Constructs a TranscriptomeTextDualEncoder processor which wraps an transcriptome processor and a tokenizer into a single
    processor.

    [`TranscriptomeTextDualEncoderProcessor`] offers all the functionalities of [`AutoTranscriptomeProcessor`] and [`AutoTokenizer`].
    See the [`~TranscriptomeTextDualEncoderProcessor.__call__`] and [`~TranscriptomeTextDualEncoderProcessor.decode`] for more
    information.

    Args:
        transcriptome_processor ([`AutoTranscriptomeProcessor`], *optional*):
            The transcriptome processor is a required input.
        tokenizer ([`PreTrainedTokenizer`], *optional*):
            The tokenizer is a required input.
    """
    attributes = ["tokenizer", "transcriptome_processor"]  # "transcriptome_processor",
    transcriptome_processor_class = "ProcessorMixin"
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        transcriptome_processor: Union[str, Any] = None,
        tokenizer: Union[str, Any] = None,
        nproc: int = 8,
        **transcriptome_kwargs,
    ):
        if transcriptome_processor == "geneformer":
            transcriptome_processor = GeneformerTranscriptomeProcessor(
                nproc=nproc,
                emb_label="natural_language_annotation",  # config["anndata_label_name"]
                **transcriptome_kwargs,
            )
        elif transcriptome_processor == "scgpt":
            transcriptome_processor = ScGPTTranscriptomeProcessor(
                nproc=nproc,
                **transcriptome_kwargs,
            )
        else:
            assert (
                transcriptome_processor is not None
            ), "You have to specify a transcriptome processor."
            transcriptome_processor = transcriptome_processor

        if isinstance(tokenizer, str):
            tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        if tokenizer is None:
            raise ValueError("You have to specify a tokenizer.")

        super().__init__(tokenizer, transcriptome_processor)

    def __call__(
        self,
        text=None,
        transcriptomes=None,
        text_truncation=True,
        return_tensors=None,
        **kwargs,
    ):
        """
        Main method to prepare for the model one or several sequences(s) and transcriptome(s). This method forwards the `text`
        and `kwargs` arguments to TranscriptomeTextDualEncoderTokenizer's [`~PreTrainedTokenizer.__call__`] if `text` is not
        `None` to encode the text. To prepare the transcriptome(s), this method forwards the `transcriptomes` and `kwargs` arguments to
        AutoTranscriptomeProcessor's [`~AutoTranscriptomeProcessor.__call__`] if `transcriptomes` is not `None`. Please refer to the doctsring
        of the above two methods for more information.

        Args:
            text (`str`, `List[str]`, `List[List[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            transcriptomes (`PIL.Transcriptome.Transcriptome`, `np.ndarray`, `torch.Tensor`, `List[PIL.Transcriptome.Transcriptome]`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The transcriptome or batch of transcriptomes to be prepared. Each transcriptome can be a PIL transcriptome, NumPy array or PyTorch
                tensor. In case of a NumPy array/PyTorch tensor, each transcriptome should be of shape (C, H, W), where C is a
                number of channels, H and W are transcriptome height and width.

            text_truncation: Whether to truncate text if the sequence is longer than the maximum length of the model. NOTE: This currently *enforces* the length to be 100 (parameter provided in `dataset/jointemb.py`). Better might be to truncate to the maximum length within the batch (if smaller than 100)

            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors of a particular framework. Acceptable values are:

                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return NumPy `np.ndarray` objects.
                - `'jax'`: Return JAX `jnp.ndarray` objects.

        Returns:
            [`BatchEncoding`]: A [`BatchEncoding`] with the following fields:

            - **input_ids** -- List of token ids to be fed to a model. Returned when `text` is not `None`.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
              `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if `text` is not
              `None`).
            - **expression_values** -- Pixel values to be fed to a model. Returned when `transcriptomes` is not `None`.
        """

        if text is None and transcriptomes is None:
            raise ValueError(
                "You have to specify either text or transcriptomes. Both cannot be none."
            )

        if text is not None:
            encoding = self.tokenizer(
                text,
                truncation=text_truncation,
                return_tensors=return_tensors,
                **kwargs,
            )

        if transcriptomes is not None:
            transcriptome_processor_results = self.transcriptome_processor(
                transcriptomes, return_tensors=return_tensors, **kwargs
            )

        if text is not None and transcriptomes is not None:
            if type(self.transcriptome_processor) == GeneformerTranscriptomeProcessor:
                encoding["expression_tokens"] = transcriptome_processor_results[
                    "expression_tokens"
                ]
                encoding["expression_token_lengths"] = transcriptome_processor_results[
                    "expression_token_lengths"
                ]
            elif type(self.transcriptome_processor) == ScGPTTranscriptomeProcessor:
                for key in transcriptome_processor_results.keys():
                    encoding[key] = transcriptome_processor_results[key]
            else:
                raise ValueError("Unsupported transcriptome processor type.")
            return encoding
        elif text is not None:
            return encoding
        else:
            return BatchEncoding(
                data=dict(**transcriptome_processor_results), tensor_type=return_tensors
            )

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to TranscriptomeTextDualEncoderTokenizer's
        [`~PreTrainedTokenizer.batch_decode`]. Please refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to TranscriptomeTextDualEncoderTokenizer's [`~PreTrainedTokenizer.decode`].
        Please refer to the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        transcriptome_processor_input_names = (
            self.transcriptome_processor.model_input_names
        )
        return list(
            dict.fromkeys(tokenizer_input_names + transcriptome_processor_input_names)
        )
