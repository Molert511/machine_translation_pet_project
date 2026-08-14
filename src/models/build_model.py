from typing import Any

import torch
import torch.nn as nn

from .rnn_based import RNNTranslationModel
from .transformer_based import TransformerTranslationModel


MODEL_CLASSES = {
    "rnn": RNNTranslationModel,
    "transformer": TransformerTranslationModel,
}


def build_model(
    config: dict[str, Any],
    device: torch.device,
    vocab_src: dict[str, int],
    vocab_tgt: dict[str, int], 
):
    model_name = config["model_name"]
    model_class = MODEL_CLASSES[model_name]

    model = model_class(device=device, vocab_src=vocab_src, vocab_tgt=vocab_tgt, **config["model"])

    return model.to(device)
