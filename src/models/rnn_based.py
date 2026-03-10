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
            rnn_type: Type = nn.LSTM,
            rnn_layers: int = 1,
            padding_idx: int = 0,
    ):
        super(RNNEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=padding_idx)
        self.rnn = rnn_type(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=rnn_layers,
            batch_first=True,
            bidirectional=True,
        )

        self.fc_h = nn.Linear(hidden_size * 2, hidden_size)
        self.fc_c = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, x, lengths):
        embeddings = self.embedding(x)
        packed_embeddings = pack_padded_sequence(embeddings, lengths.cpu(), batch_first=True, enforce_sorted=False)
        outputs, (h, c) = self.rnn(packed_embeddings)
        padded_outputs, _ = pad_packed_sequence(outputs, batch_first=True)

        h = torch.cat([h[::2], h[1::2]], dim=2)
        c = torch.cat([c[::2], c[1::2]], dim=2)
        h = torch.relu(self.fc_h(h))
        c = torch.relu(self.fc_c(c))

        return padded_outputs, (h, c)


class RNNDecoder(nn.Module):
    def __init__(
            self,
            vocab_size: int,
            embed_size: int = 256,
            hidden_size: int = 256,
            rnn_type: Type = nn.LSTM,
            rnn_layers: int = 1,
            padding_idx: int = 0,
    ):
        super(RNNDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=padding_idx)
        self.rnn = rnn_type(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=rnn_layers,
            batch_first=True,
        )

        self.fc_out = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, h_input, c_input):
        embedding = self.embedding(x)
        output, (h, c) = self.rnn(embedding, (h_input, c_input))
        prediction = self.fc_out(output.squeeze(1))

        return prediction, h, c


class RNNTranslationModel(nn.Module):
    def __init__(
            self,
            device,
            vocab_src,
            vocab_trg,
            embed_size = 256,
            hidden_size = 256,
            rnn_type = nn.LSTM,
            rnn_layers = 1,
            max_length = 100,
    ):
        super(RNNTranslationModel, self).__init__()
        self.device = device
        self.vocab_src = vocab_src
        self.vocab_trg = vocab_trg
        self.max_length = max_length

        self.encoder = RNNEncoder(
            vocab_size=len(self.vocab_src),
            embed_size=embed_size,
            hidden_size=hidden_size,
            rnn_type=rnn_type,
            rnn_layers=rnn_layers,
            padding_idx=self.vocab_src["<pad>"],
        )
        self.decoder = RNNDecoder(
            vocab_size=len(self.vocab_trg),
            embed_size=embed_size,
            hidden_size=hidden_size,
            rnn_type=rnn_type,
            rnn_layers=rnn_layers,
            padding_idx=self.vocab_trg["<pad>"],
        )

    def forward(self, src, src_lengths, trg = None):
        batch_size = src.shape[0]
        _, (h, c) = self.encoder(src, src_lengths)

        if trg is not None:
            trg_len = trg.shape[1]
            trg_vocab_size = self.decoder.fc_out.out_features
            outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
            input_token = trg[:, 0].unsqueeze(1)

            for t in range(1, trg_len):
                output, h, c = self.decoder(input_token, h, c)
                outputs[:, t] = output
                input_token = trg[:, t].unsqueeze(1)

            return outputs
        else:
            return self.translate(src, src_lengths)

    def translate(self, src, src_lengths):
        with torch.no_grad():
            batch_size = src.shape[0]
            _, (h, c) = self.encoder(src, src_lengths)
            input_token = torch.tensor([self.vocab_trg["<bos>"]] * batch_size).unsqueeze(1).to(self.device)

            translations = [[] for _ in range(batch_size)]
            is_finished = [False] * batch_size

            for _ in range(self.max_length):
                output, h, c = self.decoder(input_token, h, c)
                current_token = output.argmax(1)

                for i, token in enumerate(current_token):
                    if not is_finished[i]:
                        if token.item() == self.vocab_trg["<eos>"]:
                            is_finished[i] = True
                        else:
                            translations[i].append(token.item())
                input_token = current_token.unsqueeze(1)

                if all(is_finished):
                    break

            return translations
