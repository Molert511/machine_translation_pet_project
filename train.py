import argparse
from typing import Any
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.config_utils import load_config
from src.datasets import TextDataset
from src.models import build_model
from src.training import Trainer
from src.metrics import bleu


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    vocab_src: dict[str, int],
    vocab_tgt: dict[str, int],
    config: dict[str, Any],
    path: str,
) -> None:
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "vocab_src": vocab_src,
        "vocab_tgt": vocab_tgt,
        "config": config,
    }

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def main(config_path: str) -> None:
    config = load_config(config_path)
    device = torch.device(config["device"])

    train_dataset = TextDataset(
        config["dataset"]["src_train"],
        config["dataset"]["tgt_train"],
        max_length=config["dataset"]["max_length"],
        create_vocab=True,
    )

    if (train_dataset.vocab_src is None) or (train_dataset.vocab_tgt is None):
        raise RuntimeError("Could not create vocabs")

    val_dataset = TextDataset(
        config["dataset"]["src_val"],
        config["dataset"]["tgt_val"],
        max_length=config["dataset"]["max_length"],
        vocab_src=train_dataset.vocab_src,
        vocab_tgt=train_dataset.vocab_tgt,
        create_vocab=False,
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

    dataloaders = {
        "train": train_loader,
        "val": val_loader,
    }

    model = build_model(
        config=config,
        device=device,
        vocab_src=train_dataset.vocab_src,
        vocab_tgt=train_dataset.vocab_tgt,
    )

    epochs_cnt = config["training"]["epochs_cnt"]

    criterion = nn.CrossEntropyLoss(ignore_index=train_dataset.vocab_tgt["<pad>"])
    optimizer = AdamW(model.parameters())
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=epochs_cnt)

    trainer = Trainer(
        model=model,
        criterion=criterion,
        metric=bleu,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        device=device,
        dataloaders=dataloaders,
        epochs_cnt=epochs_cnt,
    )

    trainer.train()

    save_checkpoint(
        model=model,
        optimizer=optimizer,
        vocab_src=train_dataset.vocab_src,
        vocab_tgt=train_dataset.vocab_tgt,
        config=config,
        path=config["checkpoint_path"]
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Machine Translations Model Training")
    parser.add_argument("--config", type=str, required=True)

    args = parser.parse_args()

    main(args.config)
