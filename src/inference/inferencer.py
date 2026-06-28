import torch
from tqdm import tqdm


class Inferencer:
    def __init__(self, model, dataloader, filename="translations.txt"):
        self.model = model
        self.dataloader = dataloader
        self.filename = filename

    def inference(self):
        indices_to_words_dict = {value: key for key, value in self.model.vocab_trg.items()}

        model_translations = []

        for i, batch in enumerate(tqdm(self.dataloader, desc="Inference")):
            src_rows = batch["src_row"].to(self.model.device)
            src_lens = batch["src_len"]

            batch_translations = self.model.translate(src_rows, src_lens)

            for i, indices in enumerate(batch_translations):
                model_translation_words = [indices_to_words_dict.get(token, "<unk>") for token in indices]
                model_translations.append(" ".join(model_translation_words))

        with open(self.filename, "w") as f:
            f.write("\n".join(model_translations))
