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

from typing import Any
from typing import Union

from .wrappers import Criterion, OptimizerWrapper
from .metrics import Metric, F1_Score_Multi_Labels
from .callbacks import EarlyStopper

import time
import sys
import torch
import numpy as np


class TrainableModule(torch.nn.Module):
    """
        A base class for creating trainable PyTorch modules with convenient training and evaluation methods.

        Methods:
            - validate(self, data_loader, criterion, metrics, aggregate_on_dataset=True)
            - fit(self, train_loader, val_loader, criterion, metrics, optimizer, num_epochs, early_stopper=None,
                  aggregate_on_dataset=True, verbose=True)
            - predict(self, data, batch_size=32)

        Attributes:
            No new attributes are introduced in this class.

    """

    def __init__(self):
        super(TrainableModule, self).__init__()

    def __to_device(self,
                    data: Union[torch.tensor, dict, list],
                    device: torch.device) -> Union[torch.tensor, dict, list]:
        """
        Transfers the input data or nested data structures to the specified PyTorch device.

        :params: data : The input data or nested data structures to be transferred.
        :params: device The target PyTorch device.
        :returns: torch.Tensor or dict or list or other: The data or nested data structures transferred to the specified device.

        Notes:
        - For torch.Tensor objects, the method uses the `to` method to transfer the tensor to the specified device.
        - For dictionaries, the method recursively applies itself to each value in the dictionary.
        - For lists, the method recursively applies itself to each element in the list.
        - Other data types are returned unchanged.

        Example:
        ```python
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        data = {
            'image': torch.rand((3, 256, 256)),
            'mask': torch.tensor([0, 1, 2])
        }

        transformed_data = __to_device(data, device)
        ```

        In this example, the `image` tensor and the `mask` tensor inside the `data` dictionary are transferred to the specified device.
        """

        if isinstance(data, torch.Tensor):
            return data.to(device)
        elif isinstance(data, dict):
            return {key: self.__to_device(value, device) for key, value in data.items()}
        elif isinstance(data, list):
            return [self.__to_device(item, device) for item in data]
        else:
            return data

    def validate(self,
                 data_loader: torch.utils.data.DataLoader,
                 criterion: Criterion,
                 metrics: list[Metric],
                 aggregate_loss_on_dataset: bool = True) -> dict[str, float] | tuple[dict[str, float], torch.Tensor]:

        """
        Validate the model on the given data loader.

        :param data_loader: DataLoader containing data.
        :param criterion: A dictionary containing the name and the loss function to use.
        :param metrics: A list of dictionaries containing the name and the functions of the metrics to calculate.
        :param aggregate_loss_on_dataset: If True, the reduce strategy is applied over all the loss of the samples of the dataset.
                                          Otherwise, the reduce strategy is applied over all the batches to get a partial loss for each batch,
                                          then the partial losses are reduced to get a unique loss for the epoch.
                                          In the first case, the result is more accurate, but more RAM is used.
                                          In the second case, the result is less accurate due to numerical approximation, and less RAM is used.
        :return: A dictionary containing the validation results, including the loss value and metrics.
                 If `return_predictions` is True, returns a tuple containing the results dictionary and a tensor with
                 model predictions.
        """

        results = {criterion.name: []}
        for metric in metrics:
            results[metric.name] = []

        aggregated_losses = torch.tensor([], device='cpu')

        self.eval()
        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(data_loader):
                inputs, labels = self.__to_device(inputs, next(self.parameters()).device), labels.to(next(self.parameters()).device)
                outputs = self(inputs)
                loss = criterion(outputs, labels)

                predicted_class_id = torch.round(outputs)

                if aggregate_loss_on_dataset:
                    aggregated_losses = torch.cat((aggregated_losses, loss.to('cpu')))
                else:
                    reduced_batch_loss = criterion.reduction_function(loss)
                    aggregated_losses = torch.cat((aggregated_losses, reduced_batch_loss.unsqueeze(0).to('cpu')))

                for metric in metrics:
                    metric.update_state(predicted_class_id, labels)

        results[criterion.name] = criterion.reduction_function(aggregated_losses).item()

        for metric in metrics:
            results[metric.name] = metric.get_result()
            metric.reset_state()

        return results

    def fit(self,
            train_loader: torch.utils.data.DataLoader,
            val_loader: torch.utils.data.DataLoader,
            criterion: Criterion,
            metrics: list[Metric],
            optimizer: OptimizerWrapper,
            num_epochs: int,
            early_stopper: EarlyStopper = None,
            aggregate_loss_on_dataset: bool = True,
            verbose: bool = True) -> dict[str, dict[str, list[Any]]]:
        """
        Train the model.

        :param train_loader: Training data loader.
        :param val_loader: Validation data loader
        :param criterion: Loss function.
        :param metrics: List of dictionaries containing two fields name and function
        :param optimizer: Model optimizer.
        :param num_epochs: Number of training epochs.
        :param early_stopper: An object of type EarlyStopper, if None training goes on until num_epochs has been done
        :param aggregate_loss_on_dataset: If True, the reduce strategy is applied over all the loss of the samples of the dataset.
                                          Otherwise, the reduce strategy is applied over all the batches to get a partial loss for each batch,
                                          then the partial losses are reduced to get a unique loss for the epoch.
                                          In the first case, the result is more accurate, but more RAM is used.
                                          In the second case, the result is less accurate due to numerical approximation, and less RAM is used.
        :param verbose: If true print the training process results
        :returns: A dictionary that contain the history for loss and each metric

        """

        optimizer = optimizer.get_optimizer(self.parameters())

        train_history = {criterion.name: []}
        val_history = {criterion.name: []}
        for metric in metrics:
            train_history[metric.name] = []
            val_history[metric.name] = []
        iterations_per_epoch = len(train_loader)

        if early_stopper and early_stopper.restore_weights:
            early_stopper.create_directory()

        for epoch in range(num_epochs):
            start_time = time.time()
            self.train()

            for iteration, (inputs, labels) in enumerate(train_loader):
                inputs, labels = self.__to_device(inputs, next(self.parameters()).device), labels.to(
                    next(self.parameters()).device)
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss = criterion.reduction_function(loss)
                loss.backward()
                optimizer.step()
                metrics_value = []

                predicted_class_id = torch.round(outputs)

                for metric in metrics:
                    metrics_value.append(metric(predicted_class_id, labels))

                if verbose:
                    out_str = f"\r Epoch: {epoch + 1}/{num_epochs} Iterations: {iteration + 1}/{iterations_per_epoch} - {criterion.name}: {loss.item()}"
                    for idx, metric in enumerate(metrics):
                        out_str += f" - {metric.name}: {metrics_value[idx]}"
                    sys.stdout.write(out_str)
                    sys.stdout.flush()

            train_results = self.validate(train_loader,
                                          criterion,
                                          metrics,
                                          aggregate_loss_on_dataset=aggregate_loss_on_dataset)
            val_results = self.validate(val_loader,
                                        criterion,
                                        metrics,
                                        aggregate_loss_on_dataset=aggregate_loss_on_dataset)

            end_time = time.time()

            for key, value in train_results.items():
                train_history[key].append(value)
            for key, value in val_results.items():
                val_history[key].append(value)

            if verbose:
                out_str = f"\r Epoch: {epoch + 1}/{num_epochs} Iterations: {iterations_per_epoch}/{iterations_per_epoch} Time: {np.round(end_time - start_time, decimals=3)}s"
                for key, value in train_results.items():
                    out_str += f" - {key}: {np.round(value, 15)}"
                for key, value in val_results.items():
                    out_str += f" - {'val_' + key}: {np.round(value, 15)}"

                sys.stdout.write("\r" + " " * len(out_str) + "\r")
                sys.stdout.flush()
                sys.stdout.write(out_str)
                sys.stdout.flush()
                print()

            if early_stopper and early_stopper(val_history[early_stopper.monitor], self):
                if verbose:
                    print(early_stopper.get_message())
                break

        if early_stopper and early_stopper.restore_weights:
            early_stopper.delete_directory()
        if verbose:
            print("Train Completed")

        return {'train': train_history, 'val': val_history}

    def predict(self, data: Union[torch.Tensor, dict]) -> torch.Tensor:
        """
        Generate predictions for the given input data using the trained model.

        :param data: Input data for which predictions are to be generated.
        :return: Tensor containing the predicted labels.
        """

        predicted_labels_list = []
        for batch_data, y in data:

            batch_data_device = self.__to_device(data=batch_data, device=next(self.parameters()).device)
            batch_output = self(batch_data_device)
            batch_output = torch.round(batch_output).detach().cpu()
            predicted_labels_list.append(batch_output)

        predicted_labels = torch.cat(predicted_labels_list, dim=0)

        return predicted_labels
