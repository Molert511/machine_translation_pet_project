import matplotlib.pyplot as plt


def plot_losses_metrics(train_losses: list[float], val_losses: list[float]):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train loss")
    plt.plot(range(1, len(val_losses) + 1), val_losses, label="Val loss")
    plt.ylabel("loss")

    plt.xlabel("epoch")

    plt.show()
