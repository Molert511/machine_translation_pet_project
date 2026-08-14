from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sacrebleu.metrics import BLEU


def bleu(model: nn.Module, dataloader: DataLoader) -> float:
    model.eval()
    indices_to_words_dict = {value: key for key, value in model.vocab_tgt.items()}

    model_translations = []
    reference_translations = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Metric calculation"):
            src_rows = batch["src_row"].to(model.device)
            src_lens = batch["src_len"].tolist()
            tgt_rows = batch["tgt_row"].to(torch.device("cpu"))
            tgt_lens = batch["tgt_len"].tolist()

            batch_indices = model(src_rows, src_lens)

            for i, sentence_indices in enumerate(batch_indices):
                sentence_translation = [indices_to_words_dict.get(idx, "<unk>") for idx in sentence_indices]
                model_translations.append(" ".join(sentence_translation))

                reference_sentence_translation = [indices_to_words_dict.get(idx, "<unk>") for idx in tgt_rows[1:tgt_lens[i] + 1]]
                reference_translations.append([" ".join(reference_sentence_translation)])

        bleu_metric = BLEU()

        return bleu_metric.corpus_score(model_translations, reference_translations).score
