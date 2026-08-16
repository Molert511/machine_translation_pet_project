from typing import Protocol

from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models import ModelProtocol


class Inferencer:
    def __init__(self, model: ModelProtocol, dataloader: DataLoader, filename: str = "translations.txt") -> None:
        self.model = model
        self.dataloader = dataloader
        self.filename = filename

    def inference(self) -> None:
        self.model.eval()

        indices_to_words_dict = {value: key for key, value in self.model.vocab_tgt.items()}
        model_translations = []

        with torch.no_grad():
            for batch in tqdm(self.dataloader, desc="Inference"):
                src_rows = batch["src_row"].to(self.model.device)
                src_lens = batch["src_len"].tolist()

                batch_indices = self.model.translate(src_rows, src_lens)

                for sentence_indices in batch_indices:
                    sentence_translation = [indices_to_words_dict.get(idx, "<unk>") for idx in sentence_indices]
                    model_translations.append(" ".join(sentence_translation))

        with open(self.filename, "w") as f:
            f.write("\n".join(model_translations))
