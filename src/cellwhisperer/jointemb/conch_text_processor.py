from typing import Any, Dict, Optional, Union, List

import torch
from transformers.processing_utils import ProcessorMixin
from conch.open_clip_custom import get_tokenizer, tokenize


class ConchTextProcessor(ProcessorMixin):
    attributes = []

    def __init__(self, *args, **kwargs):
        self._tokenizer = get_tokenizer()
        super().__init__(*args, **kwargs)

    def __call__(
        self,
        text: Union[str, List[str]],
        return_tensors: Optional[str] = "pt",
        *args: Any,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        if isinstance(text, str):
            text = [text]
        input_ids = tokenize(texts=text, tokenizer=self._tokenizer)
        # Build attention mask: 1 for non-pad, 0 for pad
        pad_id = self._tokenizer.pad_token_id
        attention_mask = (input_ids != pad_id).long()
        if return_tensors == "pt":
            return {"input_ids": input_ids, "attention_mask": attention_mask}
        elif return_tensors is None:
            return {
                "input_ids": input_ids.cpu().numpy(),
                "attention_mask": attention_mask.cpu().numpy(),
            }
        else:
            raise ValueError(f"Unsupported return_tensors type: {return_tensors}. Use 'pt' or None.")

    @property
    def model_input_names(self):
        return ["input_ids", "attention_mask"]
