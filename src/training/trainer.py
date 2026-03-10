from tqdm import tqdm
import torch

from src.tools import plot_losses_metrics


class Trainer:
    def __init__(
        self,
        model,
        criterion,
        metric,
        optimizer,
        lr_scheduler,
        config,
        device,
        dataloaders,
        epochs_cnt,
    ):
        
        self.is_train = True

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.metric = metric

        self.config = config
        self.device = device

        self.train_dataloader = dataloaders["train"]
        self.val_dataloader = dataloaders["val"]

        self.epochs_cnt = epochs_cnt

    def train(self):
        train_losses = []
        val_losses = []

        val_metrics = []

        for epoch in range(self.epochs_cnt):
            print(f"Epoch: {epoch + 1}/{self.epochs_cnt}")
            train_loss = self.train_epoch()
            val_loss, val_metric = self.val_epoch()

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            val_metrics.append(val_metric)

            self.lr_scheduler.step()

            plot_losses_metrics(train_losses, val_losses, val_metrics)

        torch.save(self.model.state_dict(), "model.pt")

    def train_epoch(self):
        self.model.train()
        self.is_train = True
        total_loss = 0
        total_tokens = 0

        for i, batch in enumerate(tqdm(self.train_dataloader, desc="Train")):
            batch_loss, batch_tokens_cnt = self.process_batch(batch)
            total_loss += batch_loss
            total_tokens += batch_tokens_cnt

        return total_loss / total_tokens
    
    @torch.no_grad()
    def val_epoch(self):
        self.model.eval()
        self.is_train = False
        total_loss = 0
        total_tokens = 0

        for i, batch in enumerate(tqdm(self.val_dataloader, desc="Validation")):
            batch_loss, batch_tokens_cnt = self.process_batch(batch)
            total_loss += batch_loss
            total_tokens += batch_tokens_cnt

        metric = self.metric(self.model, self.val_dataloader)

        return total_loss / total_tokens, metric

    def process_batch(self, batch):
        src_row = batch["src_row"].to(self.device)
        src_len = batch["src_len"]
        trg_row = batch["trg_row"].to(self.device)

        output = self.model(src_row, src_len, trg_row)

        output_dim = output.shape[-1]
        output = output[:, 1:].reshape(-1, output_dim)
        trg_row_without_bos = trg_row[:, 1:].reshape(-1)

        loss = self.criterion(output, trg_row_without_bos)

        if self.is_train:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        batch_loss = loss.item()
        batch_tokens_cnt = (trg_row_without_bos != self.model.vocab_trg["<pad>"]).sum().item()

        return (batch_loss, batch_tokens_cnt)
