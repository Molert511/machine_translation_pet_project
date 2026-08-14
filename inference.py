import argparse
from typing import Any

import torch
from torch.utils.data import DataLoader

from src.datasets import TextDataset
from src.models import build_model
from src.inference import Inferencer


def load_checkpoint(checkpoint_path: str) -> dict[str, Any]:
    return torch.load(checkpoint_path)


def main(checkpoint_path: str, src_filepath: str, output_path: str) -> None:
    checkpoint = load_checkpoint(checkpoint_path=checkpoint_path)
    config = checkpoint["config"]
    device = torch.device(config["device"])

    vocab_src = checkpoint["vocab_src"]
    vocab_tgt = checkpoint["vocab_tgt"]

    model = build_model(
        config=config,
        device=device,
        vocab_src=vocab_src,
        vocab_tgt=vocab_tgt,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    test_dataset = TextDataset(
        src_filepath,
        max_length=config["dataset"]["max_length"],
        vocab_src=vocab_src,
        create_vocab=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config["dataset"]["batch_size"],
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    inferencer = Inferencer(model=model, dataloader=test_loader, filename=output_path)
    inferencer.inference()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Machine Translation Inference")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--src_filepath", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)

    args = parser.parse_args()

    main(args.checkpoint, args.src_filepath, args.output_path)
