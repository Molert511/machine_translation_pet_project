import torch
from torch.utils.data import Dataset
from collections import Counter


class TextDataset(Dataset):
    def __init__(
            self,
            src_filepath,
            trg_filepath = None,
            max_length = 100,
            vocab_src = None,
            vocab_trg = None,
            create_vocab = False,
            min_frequency=200,
    ):
        with open(src_filepath, "r") as f:
            self.src_data = [row.split() for row in f]

        self.trg_data = None
        if trg_filepath:
            with open(trg_filepath, "r") as f:
                self.trg_data = [row.split() for row in f]

        self.max_length = max_length

        self.vocab_src = vocab_src
        self.vocab_trg = vocab_trg
        if create_vocab:
            self.vocab_src = self.create_vocab(self.src_data, min_frequency=min_frequency)
            self.vocab_trg = self.create_vocab(self.trg_data, min_frequency=min_frequency)

    def create_vocab(self, data, min_frequency):
        vocab = {
            "<pad>": 0,
            "<unk>": 1,
            "<bos>": 2,
            "<eos>": 3,
        }

        counter = Counter()
        for row in data:
            counter.update(row)

        for word, frequency in counter.items():
            if frequency >= min_frequency:
                vocab[word] = len(vocab)

        return vocab

    def __len__(self):
        return len(self.src_data)

    def __getitem__(self, idx):
        src_tokens = self.src_data[idx][:self.max_length - 2]
        src_indices = [self.vocab_src["<bos>"]] + \
                      [self.vocab_src.get(token, self.vocab_src["<unk>"]) for token in src_tokens] + \
                      [self.vocab_src["<eos>"]]
        real_len = len(src_indices)
        src_indices += [self.vocab_src["<pad>"]] * (self.max_length - real_len)

        item = {
            "src_row": torch.tensor(src_indices),
            "src_len": real_len
        }

        if self.trg_data:
            trg_tokens = self.trg_data[idx][:self.max_length - 2]
            trg_indices = [self.vocab_trg["<bos>"]] + \
                          [self.vocab_trg.get(token, 1) for token in trg_tokens] + \
                          [self.vocab_trg["<eos>"]]
            real_len = len(trg_indices)
            trg_indices += [self.vocab_trg["<pad>"]] * (self.max_length - real_len)

            item["trg_row"] = torch.tensor(trg_indices)
            item["trg_len"] = real_len

        return item
