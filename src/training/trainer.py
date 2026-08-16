from collections.abc import Callable

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from src.tools import plot_losses_metrics
from src.models import ModelProtocol


class Trainer:
    def __init__(
        self,
        model: ModelProtocol,
        criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        metric: Callable[[ModelProtocol, DataLoader], float],
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
        device: torch.device,
        dataloaders: dict[str, DataLoader],
        epochs_cnt: int,
        metric_name: str = "BLEU",
    ) -> None:

        self.is_train = True

        self.model = model
        self.criterion = criterion
        self.metric = metric
        self.metric_name = metric_name
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.device = device

        self.train_dataloader = dataloaders["train"]
        self.val_dataloader = dataloaders["val"]

        self.epochs_cnt = epochs_cnt

    def train(self) -> None:
        train_losses = []
        val_losses = []
        val_metrics = []

        for epoch in range(self.epochs_cnt):
            print(f"Epoch: {epoch + 1}/{self.epochs_cnt}")
            train_losses.append(self._train_epoch())

            val_loss, val_metric = self._val_epoch()
            val_losses.append(val_loss)
            val_metrics.append(val_metric)

            self.lr_scheduler.step()

            plot_losses_metrics(train_losses, val_losses, val_metrics, metric_name=self.metric_name)

    def _train_epoch(self) -> float:
        self.model.train()
        total_loss = 0

        for batch in tqdm(self.train_dataloader, desc="Train"):
            batch_loss = self._process_batch(batch)

            self.optimizer.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += batch_loss.item()

        return total_loss / len(self.train_dataloader)

    @torch.no_grad()
    def _val_epoch(self) -> tuple[float, float]:
        self.model.eval()
        total_loss = 0

        for batch in tqdm(self.val_dataloader, desc="Validation"):
            batch_loss = self._process_batch(batch)
            total_loss += batch_loss.item()

        val_metric = self.metric(self.model, self.val_dataloader)

        return total_loss / len(self.val_dataloader), val_metric

    def _process_batch(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        src_rows = batch["src_row"].to(self.device)
        src_lens = batch["src_len"].tolist()
        tgt_rows = batch["tgt_row"].to(self.device)

        tgt_input = tgt_rows[:, :-1]
        tgt_output = tgt_rows[:, 1:]

        output = self.model(src_rows, src_lens, tgt_input)

        output = output.reshape(-1, output.shape[-1])
        tgt_output = tgt_output.reshape(-1)

        loss = self.criterion(output, tgt_output)

        return loss
