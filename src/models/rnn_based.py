import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from typing import Type


class RNNEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_size: int = 256,
        hidden_size: int = 256,
        num_layers: int = 1,
        padding_idx: int = 0,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=padding_idx)
        self.lstm = nn.LSTM(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )

        self.fc_h = nn.Linear(hidden_size * 2, hidden_size)
        self.fc_c = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, x: torch.Tensor, lengths: list[int]) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        embeddings = self.embedding(x)
        packed_embeddings = pack_padded_sequence(embeddings, lengths, batch_first=True, enforce_sorted=False)
        output, (h, c) = self.lstm(packed_embeddings)
        padded_output, _ = pad_packed_sequence(output, batch_first=True, padding_value=0.0)

        h = torch.cat([h[::2], h[1::2]], dim=2)
        c = torch.cat([c[::2], c[1::2]], dim=2)
        h = self.fc_h(h)
        c = self.fc_c(c)

        return padded_output, (h, c)


class RNNDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_size: int = 256,
        hidden_size: int = 256,
        num_layers: int = 1,
        padding_idx: int = 0,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=padding_idx)
        self.lstm = nn.LSTM(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        self.fc_out = nn.Linear(hidden_size, vocab_size)

    def forward(
        self,
        x: torch.Tensor,
        h_input: torch.Tensor,
        c_input: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        embedding = self.embedding(x)
        output, (h, c) = self.lstm(embedding, (h_input, c_input))
        prediction = self.fc_out(output)

        return prediction, (h, c)


class RNNTranslationModel(nn.Module):
    def __init__(
        self,
        device: torch.device,
        vocab_src: dict[str, int],
        vocab_tgt: dict[str, int],
        embed_size: int = 256,
        hidden_size: int = 256,
        num_layers: int = 1,
        max_length: int = 100,
    ) -> None:
        super().__init__()
        self.device = device
        self.vocab_src = vocab_src
        self.vocab_tgt = vocab_tgt
        self.max_length = max_length

        self.encoder = RNNEncoder(
            vocab_size=len(self.vocab_src),
            embed_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            padding_idx=self.vocab_src["<pad>"],
        )
        self.decoder = RNNDecoder(
            vocab_size=len(self.vocab_tgt),
            embed_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            padding_idx=self.vocab_tgt["<pad>"],
        )

    def forward(
        self,
        src: torch.Tensor,
        src_lengths: list[int],
        tgt: torch.Tensor,
    ) -> torch.Tensor:
        _, (h, c) = self.encoder(src, src_lengths)

        if tgt is not None:
            predictions, _ = self.decoder(tgt, h, c)

            return predictions

        return self.translate(h, c)

    def translate(self, src: torch.Tensor, src_lengths: list[int]) -> list[list[int]]:
        with torch.no_grad():
            _, (h, c) = self.encoder(src, src_lengths)

            batch_size = src.shape[0]
            batch_input_tokens = torch.tensor([self.vocab_tgt["<bos>"]] * batch_size).unsqueeze(1).to(self.device)

            translations = [[] for _ in range(batch_size)]
            is_finished = [False] * batch_size

            for _ in range(self.max_length):
                output, (h, c) = self.decoder(batch_input_tokens, h, c)
                current_batch_tokens = output.squeeze(1).argmax(dim=1)

                for i, token in enumerate(current_batch_tokens):
                    if not is_finished[i]:
                        if token.item() == self.vocab_tgt["<eos>"]:
                            is_finished[i] = True
                        else:
                            translations[i].append(token.item())
                batch_input_tokens = current_batch_tokens.unsqueeze(1)

                if all(is_finished):
                    break

            return translations
