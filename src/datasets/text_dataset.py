from collections import Counter
from typing import Any

import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):
    def __init__(
            self,
            src_filepath: str,
            tgt_filepath: str | None = None,
            max_length: int = 100,
            vocab_src: dict[str, int] | None = None,
            vocab_tgt: dict[str, int] | None = None,
            create_vocab: bool = False,
            min_frequency: int = 3,
    ) -> None:
        with open(src_filepath, "r") as f:
            self.src_data = [row.strip().split() for row in f]

        self.tgt_data = None
        if tgt_filepath is not None:
            with open(tgt_filepath, "r") as f:
                self.tgt_data = [row.strip().split() for row in f]

        if self.tgt_data and len(self.src_data) != len(self.tgt_data):
            raise ValueError("Source and target files have different lengths")

        self.max_length = max_length

        self.vocab_src = vocab_src
        self.vocab_tgt = vocab_tgt

        if create_vocab:
            self.vocab_src = self._create_vocab(self.src_data, min_frequency=min_frequency)
            self.vocab_tgt = self._create_vocab(self.tgt_data, min_frequency=min_frequency) if self.tgt_data is not None else None

    @staticmethod
    def _create_vocab(data: list[list[str]], min_frequency: int) -> dict[str, int]:
        vocab = {
            "<pad>": 0,
            "<unk>": 1,
            "<bos>": 2,
            "<eos>": 3,
        }

        counter = Counter()
        if data is not None:
            for row in data:
                counter.update(row)

        for word, frequency in counter.items():
            if frequency >= min_frequency:
                vocab[word] = len(vocab)

        return vocab

    def _encode_sentence(self, tokens: list[str], vocab: dict[str, int]) -> tuple[torch.Tensor, int]:
        indices = (
            [vocab["<bos>"]] +
            [vocab.get(token, vocab["<unk>"]) for token in tokens[:self.max_length - 2]] +
            [vocab["<eos>"]]
        )
        sentence_len = len(indices)
        indices += [vocab["<pad>"]] * (self.max_length - sentence_len)

        return torch.tensor(indices), sentence_len

    def __len__(self) -> int:
        return len(self.src_data)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if self.vocab_src is None:
            raise ValueError("Source vocab is empty")

        src_tokens = self.src_data[idx]
        src_indices, src_sentence_len = self._encode_sentence(src_tokens, self.vocab_src)

        item = {
            "src_row": src_indices,
            "src_len": src_sentence_len,
        }

        if (self.tgt_data is not None) and (self.vocab_tgt is not None):
            tgt_tokens = self.tgt_data[idx]
            tgt_indices, tgt_sentence_len = self._encode_sentence(tgt_tokens, self.vocab_tgt)
            item["tgt_row"] = tgt_indices
            item["tgt_len"] = tgt_sentence_len

        return item
