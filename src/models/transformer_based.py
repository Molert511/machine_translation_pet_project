import torch
import torch.nn as nn
import math

class PositionalEncoder(nn.Module):
    def __init__(self, embed_size, max_length=100):
        super(PositionalEncoder, self).__init__()

        pos_embeds = torch.zeros(max_length, embed_size)
        position = torch.arange(max_length).unsqueeze(1).float()

        denominator = torch.exp(torch.arange(0, embed_size, 2).float() * (-math.log(10000.0) / embed_size))

        pos_embeds[:, 0::2] = torch.sin(position * denominator)
        pos_embeds[:, 1::2] = torch.cos(position * denominator)

        self.register_buffer("pos_embeds", pos_embeds.unsqueeze(0))

    def forward(self, x):
        return x + self.pos_embeds[:, :x.shape[1]]


def make_decoder_mask(seq_len):
    return torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1)


class TransformerTranslationModel(nn.Module):
    def __init__(
            self,
            device,
            max_length,
            vocab_src,
            vocab_trg,
            embed_size,
            nhead,
            num_encoder_layers,
            num_decoder_layers,
            dropout,
    ):
        super(TransformerTranslationModel, self).__init__()

        self.device = device
        self.max_length = max_length
        self.vocab_src = vocab_src
        self.vocab_trg = vocab_trg
        self.embed_size = embed_size

        self.src_embeddings = nn.Embedding(len(vocab_src), embed_size, padding_idx=vocab_src["<pad>"])
        self.trg_embeddings = nn.Embedding(len(vocab_trg), embed_size, padding_idx=vocab_trg["<pad>"])

        self.positional_encoder = PositionalEncoder(embed_size=embed_size)

        self.register_buffer("trg_mask", make_decoder_mask(max_length))

        self.transformer = nn.Transformer(
            d_model=embed_size,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )

        self.fc = nn.Linear(embed_size, len(vocab_trg))

        self.fc.weight = self.trg_embeddings.weight

    def forward(self, src, src_len, trg, src_key_padding_mask, trg_key_padding_mask):

        src = self.positional_encoder(self.src_embeddings(src) * math.sqrt(self.embed_size))
        trg = self.positional_encoder(self.trg_embeddings(trg) * math.sqrt(self.embed_size))

        output = self.transformer(
            src,
            trg,
            tgt_mask=self.trg_mask[0:trg.shape[1], 0:trg.shape[1]],
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=trg_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )

        return self.fc(output)
    
    def translate(self, src, src_lengths):
        self.eval()
        with torch.no_grad():
            batch_size = src.shape[0]
            trg = torch.tensor([self.vocab_trg["<bos>"]] * batch_size).unsqueeze(1).to(self.device)

            translations = [[] for _ in range(batch_size)]
            is_finished = [False] * batch_size

            for _ in range(self.max_length):
                src_key_padding_mask = (src == self.vocab_src["<pad>"])
                trg_key_padding_mask = (trg == self.vocab_trg["<pad>"])
                output = self.forward(src, src_lengths, trg, src_key_padding_mask, trg_key_padding_mask)
                current_token = output[:, -1].argmax(-1)

                for i, token in enumerate(current_token):
                    if not is_finished[i]:
                        if token.item() == self.vocab_trg["<eos>"]:
                            is_finished[i] = True
                        else:
                            translations[i].append(token.item())
                trg = torch.cat((trg, current_token.unsqueeze(1)), dim=1)

                if all(is_finished):
                    break

            return translations
