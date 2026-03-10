from tqdm import tqdm
import torch
import sacrebleu


def bleu(model, dataloader):
    indices_to_words_dict = {value: key for key, value in model.vocab_trg.items()}

    model_translations = []
    original_translations = []

    for i, batch in enumerate(tqdm(dataloader, desc="Metric calculation")):
        src_rows = batch["src_row"].to(model.device)
        src_lens = batch["src_len"]
        trg_rows = batch["trg_row"].to(torch.device("cpu"))
        trg_lens = batch["trg_len"]

        batch_translations = model.translate(src_rows, src_lens)

        for i, indices in enumerate(batch_translations):
            model_translation_words = [indices_to_words_dict.get(token, "<unk>") for token in indices[1:src_lens[i] - 1]]
            model_translations.append(" ".join(model_translation_words))

            original_translation_words = [indices_to_words_dict.get(token, "<unk>") for token in trg_rows[1:trg_lens[i] - 1]]
            original_translations.append([" ".join(original_translation_words)])

    bleu_metric = sacrebleu.corpus_bleu(model_translations, original_translations)
    return bleu_metric.score
