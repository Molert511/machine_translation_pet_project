import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.configs import load_config
from src.datasets import TextDataset
from src.models import build_model
from src.training import Trainer
from src.metrics import bleu
from src.inference import Inferencer

def main(config_path):
    config = load_config(config_path)
    device = torch.device(config["device"])

    train_dataset = TextDataset(
        config["dataset"]["src_train"],
        config["dataset"]["trg_train"],
        max_length=config["dataset"]["max_length"],
        create_vocab=True,
    )

    val_dataset = TextDataset(
        config["dataset"]["src_val"],
        config["dataset"]["trg_val"],
        max_length=config["dataset"]["max_length"],
        vocab_src=train_dataset.vocab_src,
        vocab_trg=train_dataset.vocab_trg,
    )

    test_dataset = TextDataset(
        config["dataset"]["src_test"],
        max_length=config["dataset"]["max_length"],
        vocab_src=train_dataset.vocab_src
    )

    batch_size = config["dataset"]["batch_size"]

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    dataloaders = {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
    }

    model = build_model(
        config,
        device,
        train_dataset.vocab_src,
        train_dataset.vocab_trg,
    )

    epochs_cnt = config["training"]["epochs_cnt"]

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = AdamW(model.parameters())
    lr_scheduler = CosineAnnealingLR(optimizer, epochs_cnt)

    trainer = Trainer(
        model=model,
        criterion=criterion,
        metric=bleu,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        config=config,
        device=device,
        dataloaders=dataloaders,
        epochs_cnt=epochs_cnt,
    )

    trainer.train()

    inferencer = Inferencer(model, test_loader)
    inferencer.inference()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)

    args = parser.parse_args()

    main(args.config)
