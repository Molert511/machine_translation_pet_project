import matplotlib.pyplot as plt


def plot_losses_metrics(train_losses: list[float], val_losses: list[float], val_metrics: list[float]):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].plot(range(1, len(train_losses) + 1), train_losses, label="Train loss")
    axes[0].plot(range(1, len(val_losses) + 1), val_losses, label="Val loss")
    axes[0].set_ylabel("loss")

    axes[1].plot(range(1, len(val_metrics) + 1), val_metrics, label="BLEU metric on validation")
    axes[1].set_ylabel("perplexity")

    for ax in axes:
        ax.set_xlabel("epoch")
        ax.legend()

    plt.show()
