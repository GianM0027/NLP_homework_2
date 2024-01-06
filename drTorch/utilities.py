"""



 /$$$$$$$         /$$$$$$$$                            /$$
| $$__  $$       |__  $$__/                           | $$
| $$  \ $$  /$$$$$$ | $$  /$$$$$$   /$$$$$$   /$$$$$$$| $$$$$$$
| $$  | $$ /$$__  $$| $$ /$$__  $$ /$$__  $$ /$$_____/| $$__  $$
| $$  | $$| $$  \__/| $$| $$  \ $$| $$  \__/| $$      | $$  \ $$
| $$  | $$| $$      | $$| $$  | $$| $$      | $$      | $$  | $$
| $$$$$$$/| $$      | $$|  $$$$$$/| $$      |  $$$$$$$| $$  | $$
|_______/ |__/      |__/ \______/ |__/       \_______/|__/  |__/



"""

import torch
from matplotlib import pyplot as plt
import numpy as np


def get_std_data_loader(batch_size: int,
                        shuffle: bool,
                        data: torch.Tensor,
                        label: torch.Tensor) -> torch.utils.data.DataLoader:
    """
    Create a PyTorch DataLoader for a given dataset and labels.

    This function takes your data and corresponding labels, and wraps them into a PyTorch DataLoader, which is useful
    for batching and shuffling the data during training. The DataLoader can be used in machine learning models,
    especially for tasks like training neural networks.

    :param batch_size: The number of data samples to include in each batch.
    :param shuffle: If True, the data will be shuffled at the beginning of each epoch. Use True for training and False
                    for evaluation and testing.
    :param data: A torch.Tensor representing the input data for your model. Pay attention to the specific data type.
    :param label: A torch.Tensor representing the label data for your model. Pay attention to the specific data type.

    :return: A PyTorch DataLoader object that can be iterated over to access batches of data and labels.

    Example:
    ```
    data = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])                            here the data type is int64
    data1 = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float64)      here the data type is float64

    label = torch.tensor([0, 1, 0])
    batch_size = 2
    shuffle = True

    data_loader = get_data_loader(data, label, batch_size, shuffle)
    ```
    """
    torch_dataSet = torch.utils.data.TensorDataset(data, label)
    return torch.utils.data.DataLoader(torch_dataSet, batch_size=batch_size, shuffle=shuffle)


def plot_history(history: dict[str, list[float]], patience: int = None) -> None:
    """
    Plot training and validation history for each metric.

    Parameters:
    :param history: A dictionary containing training and validation metrics history.
    :param patience: Number of epochs for patience circle (if None, no circle will be plotted).

    Notes:
    - Assumes a dictionary structure with 'train' and 'val' keys, each containing metrics as subkeys.
    - This function is specially designed to work with the fitting function defined inside the modules.py file

    Example:
    # Assuming `history` contains your training and validation history
    plot_history(your_history_variable, patience=5)
    """
    metrics = list(history['train'].keys())  # Assuming all metrics are present in the 'train' field
    num_metrics = len(metrics)

    fig, axes = plt.subplots(num_metrics, 1, figsize=(10, 5 * num_metrics))

    for i, metric in enumerate(metrics):
        epochs = range(1, len(history['train'][metric]) + 1)
        axes[i].plot(epochs, history['train'][metric], label='Training')
        axes[i].plot(epochs, history['val'][metric], label='Validation')

        if patience is not None:
            idx_last_patience = len(history['val'][metric]) - patience
            max_value = history['val'][metric][idx_last_patience - 1]
            axes[i].scatter(idx_last_patience, max_value, color='red', marker='o', label=f'Best model obtained with patient = {patience}')

        axes[i].set_title(f'{metric} History')
        axes[i].set_xlabel('Epoch')
        axes[i].set_ylabel(metric)
        axes[i].legend()
        axes[i].grid(True)

    plt.tight_layout()
    plt.show()
