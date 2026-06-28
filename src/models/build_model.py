from .rnn_based import RNNTranslationModel
from .transformer_based import TransformerTranslationModel


MODEL_CLASSES = {
    "rnn": RNNTranslationModel,
    "transformer": TransformerTranslationModel,
}


def build_model(config, device, vocab_src, vocab_trg):
    model_name = config["model_name"]
    model_class = MODEL_CLASSES[model_name]

    model = model_class(device=device, vocab_src=vocab_src, vocab_trg=vocab_trg, **config["model"])

    return model.to(device)
