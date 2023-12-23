"""
This model emulates a real model actually just retrieves cached processed model outputs whenever possible.

NOTE: This might be more elegantly implemented within model.py (e.g. via a decorator on the get_text_features and get_transcriptome_features methods)
"""

import fcntl
import torch
import time
import torch.nn as nn
import pickle
from typing import Optional, Union
from pathlib import Path
import torch
import hashlib
import logging
from single_cellm.jointemb.scgpt_model import ScGPTConfig
from single_cellm.config import get_path

logger = logging.getLogger(__name__)


def hash_object(obj):
    if isinstance(obj, torch.Tensor):
        return hashlib.sha256(obj.cpu().numpy().tobytes()).hexdigest()
    elif isinstance(obj, list):
        return hashlib.sha256(str([hash_object(i) for i in obj]).encode()).hexdigest()
    elif isinstance(obj, dict):
        return hashlib.sha256(
            str({k: hash_object(v) for k, v in obj.items()}).encode()
        ).hexdigest()
    elif isinstance(obj, ScGPTConfig):
        obj_to_dict = obj.to_dict()
        obj_to_dict["vocab_path"] = str(
            obj_to_dict["vocab_path"]
        )  # POSIX path objects are not hashable
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

    TODO use https://docs.python.org/3/library/shelve.html instead of a dict-pickle (not that this also requires file locking upon saving!)
    """

    def __init__(self, model):
        super(FrozenCachedModel, self).__init__()
        # Avoiding parameter tracking
        self._models = [model.eval().cpu()]

        self.model_hash = hash_object(self.model.config)
        logging.info(f"Initializing frozen model with hash {self.model_hash}")
        logging.debug(f"Corresponding model config: {self.model.config}")

        self.cache_file = get_path(["paths", "model_cache"], model_hash=self.model_hash)
        self.cache = self._load_cache()

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
                except EOFError:
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
        logger.info("Saving cache")

        assert self.model_hash == hash_object(self.model.config), "Model conf changed."

        logger.info(f"Saving cache for model {self.model_hash}")
        # make sure the directory exists
        Path(self.cache_file).parent.mkdir(parents=True, exist_ok=True)
        for _ in range(3):
            try:
                with open(self.cache_file, "wb") as f:
                    fcntl.flock(f, fcntl.LOCK_EX)  # Exclusive lock
                    pickle.dump(self.cache, f)
                    fcntl.flock(f, fcntl.LOCK_UN)  # Unlock the file
                    logging.info(f"Saved cache to {self.cache_file}")
                    break
            except OSError:
                logging.info(f"File {self.cache_file} blocked. Waiting 5 seconds")
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
        Returns a tuple of tensors (token_embeddings, Optional[sentence_embedding]). The sentence_embedding is just returned if provided by the model
        """
        assert len(args) == 0, "not implemented for positional args"

        device = [x for x in kwargs.values() if isinstance(x, torch.Tensor)][0].device
        batch_size = len([x for x in kwargs.values() if isinstance(x, torch.Tensor)][0])

        # First, we need to hash the inputs per sample (not per batch)
        sample_hashes = self.compute_sample_hashes(**kwargs)

        cache_misses = sum([sh not in self.cache for sh in sample_hashes])
        logger.debug(
            f"The cache lacks {cache_misses}/{batch_size} of the batch samples."
        )

        res_0_list = []
        res_1_list = []
        has_aggregation = None

        for i in range(batch_size):
            if sample_hashes[i] in self.cache:
                cached_result = self.cache[sample_hashes[i]]
                res_0_list.append(cached_result[0].to(device=device))
                if len(cached_result) > 1 and cached_result[1] is not None:
                    res_1_list.append(cached_result[1].to(device=device))
                else:
                    res_1_list = None
            else:
                if self.model.device == torch.device("cpu"):
                    logger.warning(
                        f"Loading model {self.model.__class__.__name__} into GPU. Consider precomputing all samples first to avoid model loading."
                    )
                    self.to(device)
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
                    res_0_list.append(model_output[0][0].detach())
                    if has_aggregation:
                        res_1_list.append(
                            model_output[1][0].detach()
                            if res_1_list is not None
                            else None
                        )
                    else:
                        res_1_list = None
                    entry = (
                        (
                            model_output[0][0].detach().cpu(),
                            model_output[1][0].detach().cpu(),
                        )
                        if has_aggregation
                        else (model_output[0][0].detach().cpu(), None)
                    )
                    self.cache[sample_hashes[i]] = entry

        res_0 = torch.stack(res_0_list).to(device=device)
        res_1 = (
            torch.stack(res_1_list).to(device=device)
            if res_1_list is not None
            else None
        )
        res = (res_0, res_1)

        # Return the data in the compatible tuple format
        return res
