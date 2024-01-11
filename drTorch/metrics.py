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

from typing import Optional

from abc import ABC, abstractmethod

import torch
import numpy as np


class Metric(ABC):
    """
    Abstract base class for implementing evaluation metrics.

    Attributes:
        name (str): Name of the metric.

    Methods:
        __call__(self, predicted_classes, target_classes, accumulate_statistic=False):
            Computes the metric based on predicted and target classes.
        update_state(self, predicted_classes, target_classes):
            Updates the internal state of the metric.
        reset_state(self, *args, **kwargs):
            Resets the internal state of the metric.
        get_result(self, *args, **kwargs):
            Computes and returns the final result of the metric.
    """
    name = None

    @abstractmethod
    def __call__(self,
                 predicted_classes: torch.Tensor,
                 target_classes: torch.Tensor,
                 accumulate_statistic: bool = False
                 ):
        pass

    @abstractmethod
    def update_state(self,
                     predicted_classes: torch.Tensor,
                     target_classes: torch.Tensor):
        pass

    @abstractmethod
    def reset_state(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_result(self, *args, **kwargs):
        pass


class F1_Score(Metric):
    """
    F1 Score metric implementation for multiclass classification tasks.

      Attributes:
        name (str): Name of the metric.
        mode (str): Computation mode for F1 Score ('none', 'macro', 'micro').
        num_classes (int): Number of classes in the classification task.
        classes_to_exclude (list[int] or np.ndarray[int]): Classes to exclude from the computation.
        classes_to_consider (np.ndarray[int]): Classes to consider for computation.
        tps (np.ndarray): True positives for each class.
        fps (np.ndarray): False positives for each class.
        fns (np.ndarray): False negatives for each class.

    Methods:
        __call__(self, predicted_classes, target_classes, accumulate_statistic=False):
            Computes the F1 Score based on predicted and target classes.
        update_state(self, predicted_classes, target_classes):
            Updates the internal state of the F1 Score metric.
        reset_state(self, *args, **kwargs):
            Resets the internal state of the F1 Score metric.
        get_result(self, *args, **kwargs):
            Computes and returns the final F1 Score result.
        __str__(self) -> str:
            Returns the name of the metric as a string.
    """

    def __init__(self,
                 name: str,
                 num_classes: int,
                 mode: str = 'macro',
                 classes_to_exclude: Optional[list[int] | np.ndarray[int]] = None):
        """

        :param name: Name of the metric.
        :param num_classes:  Number of classes in the classification task.
        :param mode: Computation mode for F1 Score ('none', 'macro', 'micro').
        :param classes_to_exclude: Classes to exclude from the computation.

        """

        self.name = name
        self.mode = mode
        self.num_classes = num_classes
        self.classes_to_exclude = classes_to_exclude if classes_to_exclude else []
        self.classes_to_consider = np.arange(num_classes)[~np.isin(np.arange(num_classes), self.classes_to_exclude)]
        self.tps = np.zeros((self.num_classes,))
        self.fps = np.zeros((self.num_classes,))
        self.fns = np.zeros((self.num_classes,))

    def __call__(self,
                 predicted_classes: torch.Tensor,  # torch tensor 1-d
                 target_classes: torch.Tensor,     # torch tensor 1-d
                 accumulate_statistic: bool = False):
        """
        Compute the F1 Score based on predicted and target classes.


        :param predicted_classes: Predicted classes.
        :param target_classes: Target (ground truth) classes.
        :param accumulate_statistic: Whether to accumulate internal statistics.
        :return: Computed F1 Score.

        """

        tps, fps, fns = self.update_state(predicted_classes, target_classes)
        if not accumulate_statistic:
            self.reset_state()

        eps = np.finfo(float).eps
        denominators = 2 * tps + fps + fns
        f1s = 2 * tps / (denominators + eps)

        if self.mode == 'none':
            result = f1s[self.classes_to_consider]
        elif self.mode == 'macro':
            result = np.mean(f1s[self.classes_to_consider])
        elif self.mode == 'micro':
            result = 2 * np.sum(tps[self.classes_to_consider]) / np.sum(denominators[self.classes_to_consider])
        else:
            raise ValueError("Undefined mode specified, available modes are 'none','macro' and 'micro'")

        return result

    def __str__(self) -> str:
        """
        Get the name of the metric as a string.

        :return: Name of the metric.
        """
        return self.name

    def update_state(self,
                     predicted_classes: torch.Tensor,
                     target_classes: torch.Tensor) -> tuple[np.array, np.array, np.array]:
        """
        Update the internal state of the F1 Score metric.

        :param predicted_classes: Predicted classes.
        :param target_classes: Target (ground truth) classes.
        :return: Tuple containing true positives, false positives, and false negatives.

        """

        predicted_classes = predicted_classes.cpu().numpy()
        target_classes = target_classes.cpu().numpy()

        mask = ~np.isin(target_classes, self.classes_to_exclude)

        predicted_classes = predicted_classes[mask]
        target_classes = target_classes[mask]

        confusion_matrix = np.zeros((self.num_classes, self.num_classes))
        for predicted_id, target_id in zip(predicted_classes, target_classes):
            confusion_matrix[predicted_id, target_id] += 1

        tps = np.diag(confusion_matrix)
        fps = np.sum(confusion_matrix, axis=1) - tps
        fns = np.sum(confusion_matrix, axis=0) - tps

        self.tps += tps
        self.fps += fps
        self.fns += fns

        return tps, fps, fns

    def reset_state(self) -> None:
        """
         Reset the internal state of the F1 Score metric.

        :return: None
        """
        self.tps = np.zeros((self.num_classes,))
        self.fps = np.zeros((self.num_classes,))
        self.fns = np.zeros((self.num_classes,))

    def get_result(self) -> float:
        """
        Compute and return the final F1 Score result.

        :return: Computed F1 Score.
        """
        eps = np.finfo(float).eps
        denominators = 2 * self.tps + self.fps + self.fns
        f1s = 2 * self.tps / (denominators + eps)

        if self.mode == 'none':
            result = f1s[self.classes_to_consider]
        elif self.mode == 'macro':
            result = np.mean(f1s[self.classes_to_consider])
        elif self.mode == 'micro':
            result = 2 * np.sum(self.tps[self.classes_to_consider]) / np.sum(denominators[self.classes_to_consider])
        else:
            raise ValueError("Undefined mode specified, available modes are 'none','macro' and 'micro'")

        return result


class F1_Score_Multi_Labels:
    """
        F1_Score_Multi_Labels class implements the F1 Score metric for multi-label classification tasks.

        Args:
            name (str): Name of the metric.
            num_classes (int): Number of classes in the classification task.
            num_labels (int): Number of labels associated with each sample.
            mode (str, optional): Computation mode for F1 Score ('none', 'macro', 'micro'). Defaults to 'macro'.
            compute_mean (bool, optional): Whether to compute the mean of F1 Scores. Defaults to True.
            classes_to_exclude (list or np.ndarray, optional): Classes to exclude from the computation.

        Attributes:
            name (str): Name of the metric.
            mode (str): Computation mode for F1 Score ('none', 'macro', 'micro').
            num_classes (int): Number of classes in the classification task.
            classes_to_exclude (list or np.ndarray): Classes to exclude from the computation.
            classes_to_consider (np.ndarray): Classes to consider based on exclusion.
            num_labels (int): Number of labels associated with each sample.
            compute_mean (bool): Whether to compute the mean of F1 Scores.
            tps (np.ndarray): True positives count.
            fps (np.ndarray): False positives count.
            fns (np.ndarray): False negatives count.

        Methods:
            update_state(predicted_classes: torch.Tensor, target_classes: torch.Tensor) -> tuple[np.array, np.array, np.array]:
                Update the internal state of the F1 Score metric.
            get_result() -> float:
                Compute and return the final F1 Score result.
            reset_state() -> None:
                Reset the internal state of the F1 Score metric.
            __str__() -> str:
                Get the name of the metric as a string.
            set_mode(compute_mean: bool) -> None:
                Set the computation mode for mean.
            __call__(predicted_classes: torch.Tensor, target_classes: torch.Tensor, accumulate_statistic: bool = False) -> float:
                Update the state, compute F1 Scores, and return the result.

        Raises:
            ValueError: If an undefined mode is specified.

        Notes:
            This class is designed to works if each label has the same number of classes

        Example:
            ```python
            f1_metric = F1_Score_Multi_Labels(name='F1_Score', num_classes=10, num_labels=5)
            result = f1_metric(predicted_classes, target_classes)
            ```

        """

    def __init__(self,
                 name: str,
                 num_classes: int,
                 num_labels: int,
                 mode: str = 'macro',
                 compute_mean: bool = True,
                 classes_to_exclude: Optional[list[int] | np.ndarray[int]] = None):
        """

        :param name: Name of the metric.
        :param num_classes:  Number of classes in the classification task.
        :param num_labels: Number of labels.
        :param mode: Computation mode for F1 Score ('none', 'macro', 'micro').
        :param compute_mean: flag to compute the mean over the different labels.
        :param classes_to_exclude: Classes to exclude from the computation.

        """

        self.name = name
        self.mode = mode
        self.num_classes = num_classes
        self.classes_to_exclude = classes_to_exclude if classes_to_exclude else []
        self.classes_to_consider = np.arange(num_classes)[~np.isin(np.arange(num_classes), self.classes_to_exclude)]
        self.num_labels = num_labels
        self.compute_mean = compute_mean

        self.tps = np.zeros((self.num_labels, self.num_classes))  # (4,2)
        self.fps = np.zeros((self.num_labels, self.num_classes))
        self.fns = np.zeros((self.num_labels, self.num_classes))

    def update_state(self,
                     predicted_classes: torch.Tensor,  # (B,L)
                     target_classes: torch.Tensor) -> tuple[np.array, np.array, np.array]:
        """
        Update the internal state of the F1 Score metric.

        :param predicted_classes: Predicted classes.
        :param target_classes: Target (ground truth) classes.
        :return: Tuple containing true positives, false positives, and false negatives.

        """

        tps = np.zeros((self.num_labels, self.num_classes))  # (4, 2)
        fps = np.zeros((self.num_labels, self.num_classes))
        fns = np.zeros((self.num_labels, self.num_classes))

        for i in range(self.num_labels):
            current_pred_class = predicted_classes[:, i].cpu().detach().numpy().astype(int)
            current_target_classes = target_classes[:, i].cpu().detach().numpy().astype(int)

            mask = ~np.isin(current_target_classes, self.classes_to_exclude)

            current_pred_class = current_pred_class[mask]
            current_target_classes = current_target_classes[mask]

            confusion_matrix = np.zeros((self.num_classes, self.num_classes))
            for predicted_id, target_id in zip(current_pred_class, current_target_classes):
                confusion_matrix[predicted_id, target_id] += 1

            tps[i,:] = np.diag(confusion_matrix)
            fps[i,:] = np.sum(confusion_matrix, axis=1) - tps[i]
            fns[i,:] = np.sum(confusion_matrix, axis=0) - tps[i]

        self.tps += tps
        self.fps += fps
        self.fns += fns

        return tps, fps, fns

    def get_result(self) -> float:
        """
        Compute and return the final F1 Score result.

        :return: Computed F1 Score.
        """
        eps = np.finfo(float).eps

        denominators = 2 * self.tps + self.fps + self.fns
        f1s = 2 * self.tps / (denominators + eps)

        if self.mode == 'none':
            result = f1s[:, self.classes_to_consider]
        elif self.mode == 'macro':
            result = np.mean(f1s[:, self.classes_to_consider], axis=1)
        elif self.mode == 'micro':
            result = 2 * np.sum(self.tps[:, self.classes_to_consider], axis=1) / np.sum(
                denominators[:, self.classes_to_consider], axis=1)
        else:
            raise ValueError("Undefined mode specified, available modes are 'none','macro' and 'micro'")

        if self.compute_mean:
            result = np.mean(result)

        return result

    def reset_state(self) -> None:
        """
         Reset the internal state of the F1 Score metric.

        :return: None
        """
        self.tps = np.zeros((self.num_labels, self.num_classes))
        self.fps = np.zeros((self.num_labels, self.num_classes))
        self.fns = np.zeros((self.num_labels, self.num_classes))

    def __str__(self) -> str:
        """
        Get the name of the metric as a string.

        :return: Name of the metric.
        """
        return self.name

    def set_mode(self, compute_mean_flag: bool):
        self.compute_mean = compute_mean_flag

    def __call__(self,
                 predicted_classes: torch.Tensor,
                 target_classes: torch.Tensor,
                 accumulate_statistic: bool = False):

        """
        Compute for each label the F1 Score based on predicted and target classes.


        :param predicted_classes: Predicted classes.
        :param target_classes: Target (ground truth) classes.
        :param accumulate_statistic: Whether to accumulate internal statistics.
        :return: Computed F1 Score.

        """

        tps, fps, fns = self.update_state(predicted_classes, target_classes)
        if not accumulate_statistic:
            self.reset_state()

        eps = np.finfo(float).eps
        denominators = 2 * tps + fps + fns
        f1s = 2 * tps / (denominators + eps)

        if self.mode == 'none':
            result = f1s[:, self.classes_to_consider]
        elif self.mode == 'macro':
            result = np.mean(f1s[:, self.classes_to_consider], axis=1)
        elif self.mode == 'micro':
            result = 2 * np.sum(tps[:, self.classes_to_consider], axis=1) / np.sum(
                denominators[:, self.classes_to_consider], axis=1)
        else:
            raise ValueError("Undefined mode specified, available modes are 'none','macro' and 'micro'")

        if self.compute_mean:
            result = np.mean(result)

        return result
