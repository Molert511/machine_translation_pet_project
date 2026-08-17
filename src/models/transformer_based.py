from typing import cast
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float, max_length: int) -> None:
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)
        pos_embeds: torch.Tensor = torch.zeros(max_length, d_model)
        position = torch.arange(max_length).unsqueeze(1)

        denominator = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pos_embeds[:, 0::2] = torch.sin(position * denominator)
        pos_embeds[:, 1::2] = torch.cos(position * denominator)

        self.register_buffer("pos_embeds", pos_embeds.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pos_embeds = cast(torch.Tensor, self.pos_embeds)

        return self.dropout(x + pos_embeds[:, :x.shape[1]])


def make_subsequent_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    subsequent_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).type(torch.uint8).to(device)
    return subsequent_mask == 0


def make_padding_mask(lengths: list[int], max_length: int, device: torch.device) -> torch.Tensor:
    lengths_tensor = torch.tensor(lengths).to(device) + 1
    mask = torch.arange(max_length, device=device) <= lengths_tensor.unsqueeze(1)

    return mask.unsqueeze(1).unsqueeze(2)


class MultiHeadedAttention(nn.Module):
    def __init__(self, nhead: int, d_model: int, dropout: float) -> None:
        super().__init__()
        assert d_model % nhead == 0

        self.d = d_model // nhead
        self.nhead = nhead
        self.linears = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(4)])
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size = query.shape[0]

        query, key, value = [
            lin(x).view(batch_size, -1, self.nhead, self.d).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]

        x = self._attention(query, key, value, mask=mask)

        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.nhead * self.d)
        )

        return self.linears[-1](x)

    def _attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        scores = query @ key.transpose(-2, -1) / math.sqrt(self.d)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        normalized_scores = self.dropout(F.softmax(scores, dim=-1))

        return normalized_scores @ value


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.fc_1 = nn.Linear(d_model, d_ff)
        self.fc_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc_2(self.dropout(F.relu(self.fc_1(x))))


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        nhead: int,
        d_model: int,
        d_ff: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.self_attn = MultiHeadedAttention(nhead, d_model, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(p=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, mask)))
        x = self.norm2(x + self.dropout(self.feed_forward(x)))

        return x


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        nhead: int,
        d_model: int,
        d_ff: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.self_attn = MultiHeadedAttention(nhead, d_model, dropout)
        self.cross_attn = MultiHeadedAttention(nhead, d_model, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(p=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, memory: torch.Tensor, src_mask: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, tgt_mask)))
        x = self.norm2(x + self.dropout(self.cross_attn(x, memory, memory, src_mask)))
        x = self.norm3(x + self.dropout(self.feed_forward(x)))

        return x


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        nhead: int,
        d_model: int,
        d_ff: int,
        num_encoder_layers: int,
        padding_idx: int,
        dropout: float,
        max_length: int,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.positional_encoding = PositionalEncoding(d_model, dropout, max_length)

        self.transformer_blocks = nn.ModuleList([
            TransformerEncoderLayer(nhead, d_model, d_ff, dropout)
            for _ in range(num_encoder_layers)
        ])

    def forward(self, x: torch.Tensor, lengths: list[int]) -> torch.Tensor:
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)

        mask = make_padding_mask(lengths, x.shape[1], x.device)

        for block in self.transformer_blocks:
            x = block(x, mask)

        return x


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        nhead: int,
        d_model: int,
        d_ff: int,
        num_decoder_layers: int,
        padding_idx: int,
        dropout: float,
        max_length: int,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.positional_encoding = PositionalEncoding(d_model, dropout, max_length)

        self.transformer_blocks = nn.ModuleList([
            TransformerDecoderLayer(nhead, d_model, d_ff, dropout)
            for _ in range(num_decoder_layers)
        ])

        self.fc = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        lengths: list[int]
    ) -> torch.Tensor:
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)

        src_mask = make_padding_mask(lengths, memory.shape[1], x.device)
        tgt_mask = make_subsequent_mask(x.shape[1], x.device)

        for block in self.transformer_blocks:
            x = block(x, memory, src_mask, tgt_mask)

        return self.fc(x)


class TransformerTranslationModel(nn.Module):
    def __init__(
        self,
        device: torch.device,
        vocab_src: dict[str, int],
        vocab_tgt: dict[str, int],
        d_model: int = 512,
        d_ff: int = 2048,
        num_encoder_layers: int = 1,
        num_decoder_layers: int = 1,
        nhead: int = 1,
        dropout: float = 0.1,
        max_length: int = 100,
    ) -> None:
        super().__init__()

        self.device = device
        self.vocab_src = vocab_src
        self.vocab_tgt = vocab_tgt
        self.max_length = max_length

        self.encoder = TransformerEncoder(
            vocab_size=len(vocab_src),
            nhead=nhead,
            d_model=d_model,
            d_ff=d_ff,
            num_encoder_layers=num_encoder_layers,
            padding_idx=vocab_src["<pad>"],
            dropout=dropout,
            max_length=max_length,
        )
        self.decoder = TransformerDecoder(
            vocab_size=len(vocab_tgt),
            nhead=nhead,
            d_model=d_model,
            d_ff=d_ff,
            num_decoder_layers=num_decoder_layers,
            padding_idx=vocab_tgt["<pad>"],
            dropout=dropout,
            max_length=max_length,
        )

    def forward(
        self,
        src: torch.Tensor,
        src_lengths: list[int],
        tgt: torch.Tensor,
    ) -> torch.Tensor:
        memory = self.encoder(src, src_lengths)

        return self.decoder(tgt, memory, src_lengths) 

    def translate(self, src: torch.Tensor, src_lengths: list[int]) -> list[list[int]]:
        self.eval()
        with torch.no_grad():
            batch_size = src.shape[0]
            batch_input_tokens = torch.tensor([self.vocab_tgt["<bos>"]] * batch_size).unsqueeze(1).to(self.device)

            translations = [[] for _ in range(batch_size)]
            is_finished = [False] * batch_size

            memory = self.encoder(src, src_lengths)

            for _ in range(self.max_length):
                output = self.decoder(batch_input_tokens, memory, src_lengths) 
                current_batch_tokens = output[:, -1].argmax(-1)

                for i, token in enumerate(current_batch_tokens):
                    if not is_finished[i]:
                        if token.item() == self.vocab_tgt["<eos>"]:
                            is_finished[i] = True
                        else:
                            translations[i].append(token.item()) 
                batch_input_tokens = torch.cat((batch_input_tokens, current_batch_tokens.unsqueeze(1)), dim=1)

                if all(is_finished):
                    break

            return translations
