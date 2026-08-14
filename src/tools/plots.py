import matplotlib.pyplot as plt


def plot_losses_metrics(
    train_losses: list[float],
    val_losses: list[float],
    val_metrics: list[float], 
    metric_name: str = "BLEU",
) -> None:
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(train_losses) + 1)

    ax1.plot(epochs, train_losses, label="Train loss", linestyle="-")
    ax1.plot(epochs, val_losses, label="Val loss", linestyle="--")

    ax1.set_ylabel("loss")
    ax1.set_xlabel("epoch")
    ax1.title("Train and Val losses")
    ax1.legend()
    ax1.grid()

    ax2.plot(epochs, val_metrics, label="Val Metric")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel(f"{metric_name}")
    ax2.set_title(f"Validation Metric {metric_name}")
    ax2.legend()
    ax2.grid()

    plt.tight_layout()
    plt.show()
