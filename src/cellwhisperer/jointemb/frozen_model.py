"""
This frozen-model emulates a real model, but actually just retrieves cached processed model outputs whenever possible (for efficiency).
NOTE: This might be more elegantly implemented within model.py (e.g. via a decorator on the get_text_features and get_transcriptome_features methods)
NOTE: https://docs.python.org/3/library/shelve.html would be more efficient than dict-pickle (would still require file-locking though)
"""

import fcntl
import torch
import time
import torch.nn as nn
import pickle
from typing import Optional, Union, Any
import types
from pathlib import Path
import torch
import hashlib
import logging
from cellwhisperer.config import get_cache_dir
from transformers.configuration_utils import PretrainedConfig


from transformers.configuration_utils import PretrainedConfig

logger = logging.getLogger(__name__)


def hash_object(obj):
    if isinstance(obj, torch.Tensor):
        return hashlib.sha256(obj.cpu().detach().numpy().tobytes()).hexdigest()
    elif isinstance(obj, (list, types.GeneratorType)):
        return hashlib.sha256(str([hash_object(i) for i in obj]).encode()).hexdigest()
    elif isinstance(obj, dict):
        return hashlib.sha256(
            str({k: hash_object(v) for k, v in obj.items()}).encode()
        ).hexdigest()
    elif isinstance(
        obj, PretrainedConfig
    ):  # need to convert unhashable Path for ScGPTConfig
        obj_to_dict = obj.to_dict()
        return hashlib.sha256(
            str({k: hash_object(v) for k, v in obj_to_dict.items()}).encode()
        ).hexdigest()
    else:
        return hashlib.sha256(str(obj).encode()).hexdigest()


class FrozenCachedModel(nn.Module):
    """
    This "model" emulates a real model but just retrieves cached outputs, whenever possible. This is achieved through the following measures:

    - Storing `model` in a list to avoid its property-tracking. And moving to CPU to not waste GPU memory
    - Use python `property`s to mimic the interface of the model
    - Use hashes and a python dict to store the cached model outputs

    """

    def __init__(self, model: nn.Module, use_cache: bool = True):
        super(FrozenCachedModel, self).__init__()
        # Avoiding parameter tracking and moving to CPU for memory reduction
        self._models = [model.eval().cpu()]

        if use_cache:

            self.model_hash = hash_object(self.model.parameters())
            logger.info(f"Initializing frozen model with hash {self.model_hash}")
            logger.debug(f"Corresponding model config: {self.model.config}")

            self.cache_file = Path(get_cache_dir()) / f"{self.model_hash}.pkl"
            self.cache = self._load_cache()
        else:
            self.cache = None

    @property
    def model(self):
        return self._models[0]

    @model.setter
    def model(self, value):
        self._models[0] = value  # Set model through property

    def to(self, *args, **kwargs):
        super(FrozenCachedModel, self).to(*args, **kwargs)
        self._models[0].to(*args, **kwargs)
        return self

    @property
    def config(self):
        return self.model.config

    @config.setter
    def config(self, value):
        self.model.config = value

    def _load_cache(self):
        """
        Try to load the cache file and if it does not exist, return an empty cache
        """
        cache = {}
        logger.info(f"Loading cache: {self.cache_file}")
        if self.cache_file is not None:
            for _ in range(3):
                try:
                    with open(self.cache_file, "rb") as f:
                        fcntl.flock(f, fcntl.LOCK_EX)  # Exclusive lock
                        cache = pickle.load(f)
                        fcntl.flock(f, fcntl.LOCK_UN)  # Unlock the file
                    break
                except FileNotFoundError:
                    logger.warning("Unable to find cache file. Creating new one")
                    break
                except (EOFError, pickle.UnpicklingError):
                    logger.error("Unable to load cache (EOFError). Resetting.")
                    continue  # maybe it works next time
                except OSError:
                    logger.info("Cache file in use (R/W) by other process. Waiting")
                    time.sleep(5)
                    continue
            else:
                logger.error(
                    "Unable to load cache (tried 3 times, but always locked). Resetting."
                )

        return cache

    def save_cache(self):
        if self.cache_file is None:
            return
        logger.info(f"Saving cache: {self.cache_file}")

        assert self.model_hash == hash_object(
            self.model.parameters()
        ), "Model parameters changed. This should not happen for a frozen model"

        logger.info(f"Saving cache for model {self.model_hash}")
        # make sure the directory exists
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        for _ in range(3):
            try:
                with open(self.cache_file, "wb") as f:
                    fcntl.flock(f, fcntl.LOCK_EX)  # Exclusive lock
                    pickle.dump(self.cache, f)
                    fcntl.flock(f, fcntl.LOCK_UN)  # Unlock the file
                    logger.info(f"Saved cache to {self.cache_file}")
                    break
            except OSError:
                logger.info(f"File {self.cache_file} blocked. Waiting 5 seconds")
                time.sleep(5)  # wait 5 seconds before trying again
        else:
            logger.error("Unable to save cache. Aborting.")

    def compute_sample_hashes(self, **kwargs):
        batch_size = len([x for x in kwargs.values() if isinstance(x, torch.Tensor)][0])
        # batch_size = len(next(iter(kwargs.values())))
        sample_hashes = []
        for i in range(batch_size):
            sample_kwargs = {
                k: v[i] if hasattr(v, "__getitem__") else v for k, v in kwargs.items()
            }
            sample_kwargs_hash = hash_object(sample_kwargs)
            sample_hashes.append(sample_kwargs_hash)

        return sample_hashes

    def forward(self, *args, **kwargs):
        """

        If the model returns a tuple, then the first element is assumed to be the token embeddings and the second element the sentence embeddings. Then the token embeddings are discarded

        If the model only returns a single tensor, then the sentence embeddings are assumed to be None.


        Returns a tuple of tensors (token_embeddings, Optional[sentence_embedding]). The sentence_embedding is just returned if provided by the model
        """
        if self.cache is None:
            return self.model(*args, **kwargs)

        assert len(args) == 0, "not implemented for positional args"

        active_device = [x for x in kwargs.values() if isinstance(x, torch.Tensor)][
            0
        ].device
        batch_size = len([x for x in kwargs.values() if isinstance(x, torch.Tensor)][0])

        # First, we need to hash the inputs per sample (not per batch)
        sample_hashes = self.compute_sample_hashes(**kwargs)

        cache_misses = sum([sh not in self.cache for sh in sample_hashes])
        logger.debug(
            f"The cache lacks {cache_misses}/{batch_size} of the batch samples."
        )
        if cache_misses == 0 and self.model.device == active_device:
            logger.warning(
                f"Model {self.model.__class__.__name__} is loaded on GPU, but all samples are cached in this batch. Moving model back to CPU."
            )
            self.to(device=torch.device("cpu"))

        if cache_misses > 0 and self.model.device == torch.device("cpu"):
            logger.warning(
                f"Loading model {self.model.__class__.__name__} into GPU. Consider precomputing all samples first to avoid model loading."
            )
            self.to(active_device)

        res_0_list = []
        res_1_list = []
        has_aggregation = None

        # NOTE this is much more complicated than necessary
        for i in range(batch_size):
            if sample_hashes[i] in self.cache:
                cached_result = self.cache[sample_hashes[i]]
                assert (
                    len(cached_result) == 2
                ), f"Cached result has wrong length {len(cached_result)}"
                res_0_list.append(cached_result[0])
                res_1_list.append(cached_result[1])
            else:
                with torch.no_grad():
                    model_output = self.model(
                        **{
                            k: v[i : i + 1] if hasattr(v, "__getitem__") else v
                            for k, v in kwargs.items()
                        }
                    )
                    if has_aggregation is None:
                        has_aggregation = len(model_output) > 1 and isinstance(
                            model_output[1], torch.Tensor
                        )

                    # only store what's necessary in the cache
                    if has_aggregation:
                        entry = (None, model_output[1][0].detach().cpu())
                    else:
                        entry = (model_output[0][0].detach().cpu(), None)
                    res_0_list.append(entry[0])
                    res_1_list.append(entry[1])

                    self.cache[sample_hashes[i]] = entry

        res_0 = (
            torch.stack(res_0_list).to(device=active_device)
            if res_0_list is not None
            and len(res_0_list) > 0
            and res_0_list[0] is not None
            else None
        )

        res_1 = (
            torch.stack(res_1_list).to(device=active_device)
            if res_1_list is not None
            and len(res_1_list) > 0
            and res_1_list[0] is not None
            else None
        )
        res = (res_0, res_1)

        # Return the data in the compatible tuple format
        return res
