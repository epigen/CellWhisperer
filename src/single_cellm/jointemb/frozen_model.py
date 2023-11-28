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
    else:
        return hashlib.sha256(str(obj).encode()).hexdigest()


class FrozenCachedModel(nn.Module):
    """
    This "model" emulates a real model but just retrieves cached outputs, whenever possible. This is achieved through the following measures:

    - Storing `model` in a list to avoid its property-tracking. And moving to CPU to not waste GPU memory
    - Use python `property`s to mimic the interface of the model
    - Use hashes and a python dict to store the cached model outputs
    """

    def __init__(self, model, cache_file: Optional[Union[str, Path]] = None):
        super(FrozenCachedModel, self).__init__()
        # Avoiding parameter tracking
        self._models = [model.eval().cpu()]
        self.cache_file = cache_file

        self.model_hash = hash_object(self.model.config)
        self.cache = self._load_cache()[self.model_hash]

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
        cache = {self.model_hash: {}}
        if self.cache_file is not None:
            for _ in range(3):
                try:
                    with open(self.cache_file, "rb") as f:
                        fcntl.flock(f, fcntl.LOCK_EX)  # Exclusive lock
                        cache = pickle.load(f)
                        fcntl.flock(f, fcntl.LOCK_UN)  # Unlock the file
                    if self.model_hash not in cache:
                        cache[self.model_hash] = {}
                    break
                except FileNotFoundError:
                    logger.warning("Unable to find cache file. Resetting cache.")
                    break
                except EOFError:
                    logger.error("Unable to load cache (EOFError). Resetting.")
                    continue  # maybe it works next time
                except OSError:
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

        # Reload the cache to make sure that we don't overwrite any changes from other runs
        cache = self._load_cache()
        cache[self.model_hash] = self.cache

        logger.info(f"Saving cache for model {self.model_hash} to cache.")
        # make sure the directory exists
        Path(self.cache_file).parent.mkdir(parents=True, exist_ok=True)
        for _ in range(3):
            try:
                with open(self.cache_file, "wb") as f:
                    fcntl.flock(f, fcntl.LOCK_EX)  # Exclusive lock
                    pickle.dump(cache, f)
                    fcntl.flock(f, fcntl.LOCK_UN)  # Unlock the file
                    logging.info(f"Saved cache to {self.cache_file}")
                    break
            except OSError:
                time.sleep(5)  # wait 5 seconds before trying again
        else:
            logger.error("Unable to save cache. Aborting.")

    def compute_sample_hashes(self, **kwargs):
        batch_size = len(next(iter(kwargs.values())))
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
        device = next(iter(kwargs.values())).device

        assert len(args) == 0, "not implemented for positional args"

        # First, we need to hash the inputs per sample (not per batch)
        batch_size = len(next(iter(kwargs.values())))
        sample_hashes = self.compute_sample_hashes(**kwargs)

        cache_misses = sum([sh not in self.cache for sh in sample_hashes])

        if cache_misses == 0:
            logger.debug(f"Loading batch from cache {cache_misses}/{batch_size}")
            with torch.no_grad():
                res_0 = torch.stack(
                    [self.cache[sample_hashes[i]][0] for i in range(batch_size)]
                ).to(device=device)
                res_1 = (
                    torch.stack(
                        [self.cache[sample_hashes[i]][1] for i in range(batch_size)]
                    ).to(device=device)
                    if len(self.cache[sample_hashes[0]]) > 1
                    and self.cache[sample_hashes[0]][1] is not None
                    else None
                )
            res = (res_0, res_1)
        else:
            logger.debug(f"Computing batch with model {cache_misses}/{batch_size}")
            self.to(device)  # make sure that model is on the correct device
            with torch.no_grad():
                res = self.model(*args, **kwargs)

            # some models provide an aggregation right away
            has_aggregation = len(res) > 1 and isinstance(res[1], torch.Tensor)

            for i in range(batch_size):
                entry = (
                    (res[0][i].detach().cpu(), res[1][i].detach().cpu())
                    if has_aggregation
                    else (res[0][i].detach().cpu(), None)
                )
                self.cache[sample_hashes[i]] = entry

            if cache_misses < batch_size:
                logger.warning(
                    f"The cache contains some ({cache_misses}) but not all ({batch_size}) of the batch samples. Consider precomputing all samples first to avoid model loading."
                )

        # Return the data in the compatible tuple format
        return res
