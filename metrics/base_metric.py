
import numpy as np
from abc import ABC
from utils.register import registry


"""
This module, `base_metric.py`, defines the abstract base class for metrics used in the FNLP (Federated Natural Language Processing) project. It provides a common interface and shared functionality for all metric classes, ensuring consistency and reusability across different metric implementations.

Key Components:
- `BaseMetric`: An abstract base class that defines the structure and common methods for all metrics. It includes initialization of common attributes, abstract methods for calculating and updating metrics, and utility methods for decoding tensors and logging.

Attributes:
- `tokenizer`: A tokenizer instance used for decoding tensors.
- `is_decreased_valid_metric`: A boolean indicating if a lower metric value is better.
- `best_valid_metric`: The best validation metric value observed, initialized based on `is_decreased_valid_metric`.
- `results`: A dictionary to store metric results.
- `best`: A boolean indicating if the current metric is the best observed.
- `save_outputs`: A boolean indicating if the outputs should be saved.
- `logger`: A logger instance for logging metric-related information.

Methods:
- `calculate_metric(*args)`: An abstract method to be implemented by subclasses for calculating the metric.
- `update_metrics(*args)`: An abstract method to be implemented by subclasses for updating the metric.
- `is_best`: A property to check if the current metric is the best observed.
- `best_metric`: A property to get the best metric results.
- `metric_name`: An abstract property to be implemented by subclasses to return the metric name.
- `decoded(tensor)`: A method to decode tensors using the tokenizer, replacing padding tokens with the tokenizer's pad token ID.

This module serves as the foundation for all metric implementations in the FNLP project, providing essential functionalities and enforcing a consistent interface for metric calculation and management.
"""


class BaseMetric(ABC):
    """
    Abstract base class for defining metrics in the FNLP (Federated Natural Language Processing) project.

    This class provides a common interface and shared functionality for all metric classes, ensuring consistency
    and reusability across different metric implementations.

    Attributes:
        tokenizer: A tokenizer instance used for decoding tensors.
        is_decreased_valid_metric (bool): Indicates if a lower metric value is better.
        best_valid_metric (float): The best validation metric value observed, initialized based on `is_decreased_valid_metric`.
        results (dict): A dictionary to store metric results.
        best (bool): Indicates if the current metric is the best observed.
        save_outputs (bool): Indicates if the outputs should be saved.
        logger: A logger instance for logging metric-related information.
    """
    def __init__(self, tokenizer, is_decreased_valid_metric=False, save_outputs=False):
        super().__init__()

        self.tokenizer = tokenizer
        self.is_decreased_valid_metric = is_decreased_valid_metric
        self.best_valid_metric = float("inf") if self.is_decreased_valid_metric else -float("inf")
        self.results = {}
        self.best = False
        self.save_outputs = save_outputs
        self.logger = registry.get("logger")

    def calculate_metric(self, *args):
        """
        Calculate the metric based on the given arguments.

        Args:
            *args: Variable length argument list for calculating the metric.

        Returns:
            The calculated metric value.

        """
        raise NotImplementedError

    def update_metrics(self, *args):
        """
        Update the metric based on the given arguments.

        Args:
            *args: Variable length argument list for updating the metric.

        Returns:
            None

        """
        raise NotImplementedError

    @property
    def is_best(self):
        """
        Check if the current metric is the best observed.

        Returns:
            A boolean indicating if the current metric is the best observed.

        """
        return self.best

    @property
    def best_metric(self):
        """
        Get the best metric results.

        Returns:
            A dictionary containing the best metric results.

        """
        return self.results

    @property
    def metric_name(self):
        """
        Get the metric name.

        Returns:
            The metric name.

        """
        raise NotImplementedError

    def decoded(self, tensor):
        """
        Decode tensors using the tokenizer, replacing padding tokens with the tokenizer's pad token ID.

        Args:
            tensor: A tensor to decode using the tokenizer.

        Returns:
            A list of decoded tokens.

        """
        tensor = np.where(tensor != -100, tensor, self.tokenizer.pad_token_id)
        decoded_tensor = self.tokenizer.batch_decode(tensor, skip_special_tokens=True)
        return decoded_tensor

