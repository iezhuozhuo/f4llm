import numpy as np
import pandas as pd

from metrics.base_metric import BaseMetric
from utils.general import pickle_write
from utils.register import registry
from sklearn.metrics import accuracy_score


"""
This module, `metric.py`, implements specific metric classes for the FNLP (Federated Natural Language Processing) project. These classes extend the `BaseMetric` class and provide concrete implementations for calculating and updating metrics based on evaluation predictions.

Key Components:
- `MetricForOpenAI`: A metric class for evaluating OpenAI models. It calculates metrics based on evaluation predictions and optionally saves the outputs for further analysis.
- `MetricForDPOPairwise`: A metric class for evaluating pairwise comparisons in DPO (Direct Preference Optimization) tasks. It calculates accuracy based on evaluation predictions and optionally saves the outputs.

Classes:
- `MetricForOpenAI`: Inherits from `BaseMetric` and implements methods for calculating metrics specific to OpenAI models.
- `MetricForDPOPairwise`: Inherits from `BaseMetric` and implements methods for calculating accuracy in pairwise comparison tasks.

Usage:
These metric classes are registered with the `registry` and can be used in the FNLP project to evaluate model performance. The metrics are calculated based on the predictions and labels provided during evaluation, and the results are stored for further analysis.

Example:
    from metrics.metric import MetricForOpenAI, MetricForDPOPairwise

    # Initialize the metric with a tokenizer and other parameters
    metric = MetricForOpenAI(tokenizer, is_decreased_valid_metric=False, save_outputs=True)

    # Calculate the metric based on evaluation predictions
    results = metric.calculate_metric(eval_preds)
"""


@registry.register_metric("openai")
class MetricForOpenAI(BaseMetric):
    """
    A metric class for evaluating OpenAI models in the FNLP (Federated Natural Language Processing) project.

    This class extends the `BaseMetric` class and provides a concrete implementation for calculating and updating
    metrics based on evaluation predictions. It also supports saving the outputs for further analysis.

    Attributes:
        tokenizer: A tokenizer instance used for decoding tensors.
        is_decreased_valid_metric (bool): Indicates if a lower metric value is better.
        save_outputs (bool): Indicates if the outputs should be saved.

    """
    def __init__(self, tokenizer, is_decreased_valid_metric, save_outputs):
        super().__init__(tokenizer, is_decreased_valid_metric, save_outputs)

    def calculate_metric(self, eval_preds):
        """
        Calculate the metric based on the evaluation predictions.

        Args:
            eval_preds: The evaluation predictions containing the model outputs and labels.

        Returns:
            results (dict): A dictionary containing the metric results.

        """
        # save output for test openai
        if self.save_outputs:
            preds, labels, inputs = eval_preds
        else:
            preds, labels = eval_preds
            inputs = None

        if isinstance(preds, tuple):
            preds = preds[0]

        decoded_preds = self.decoded(preds)
        decoded_labels = self.decoded(labels)
        decoded_inputs = self.decoded(inputs) if self.save_outputs else None

        checkpoint_opt_file = registry.get("checkpoint_opt_file")
        save_data = {"preds": decoded_preds, "labels": decoded_labels, "inputs": decoded_inputs}
        pickle_write(save_data, checkpoint_opt_file)

        results = {"result": {self.metric_name: 0.0}}
        return results

    def decoded(self, tensor):
        """
        Decode the tensor using the tokenizer.

        Args:
            tensor: The tensor to be decoded.

        Returns:
            decoded_tensor: The decoded tensor as a list of strings.

        """
        tensor = np.where(tensor != -100, tensor, self.tokenizer.pad_token_id)
        decoded_tensor = self.tokenizer.batch_decode(tensor, skip_special_tokens=True)
        return decoded_tensor

    @property
    def metric_name(self):
        """
        Get the metric name.

        Returns:
            metric_name (str): The name of the metric.

        """
        return "metric"


@registry.register_metric("pairwise")
class MetricForDPOPairwise(BaseMetric):
    """
    A metric class for evaluating pairwise comparisons in DPO (Direct Preference Optimization) tasks
    in the FNLP (Federated Natural Language Processing) project.

    This class extends the `BaseMetric` class and provides a concrete implementation for calculating
    and updating accuracy metrics based on evaluation predictions. It also supports saving the outputs
    for further analysis.

    """
    def __init__(self, tokenizer, is_decreased_valid_metric, save_outputs):
        super().__init__(tokenizer, is_decreased_valid_metric, save_outputs)

    def calculate_metric(self, eval_preds):
        """
        Calculate the accuracy metric based on the evaluation predictions.

        Args:
            eval_preds: The evaluation predictions containing the model outputs and labels.

        Returns:
            results (dict): A dictionary containing the metric results.

        """
        try:
            predictions = eval_preds.predictions
            preds = np.argmax(predictions, axis=1).reshape(-1)
        except:
            preds = eval_preds["preds"]

        labels = np.zeros(preds.shape)
        inputs = self.decoded(eval_preds.inputs) if self.save_outputs else None

        checkpoint_opt_file = registry.get("checkpoint_opt_file")
        save_data = {"preds": preds, "labels": labels, "inputs": inputs}
        pickle_write(save_data, checkpoint_opt_file)

        accuracy = float(accuracy_score(labels, preds, normalize=True))
        results = {"result": {self.metric_name: round(accuracy, 3)}}

        return results

    @property
    def metric_name(self):
        """
        Get the metric name.

        Returns:
            metric_name (str): The name of the metric.

        """
        return "accuracy"


@registry.register_metric("rwben")
class MetricForRewardBenwise(BaseMetric):
    def __init__(self, tokenizer, is_decreased_valid_metric, save_outputs):
        super().__init__(tokenizer, is_decreased_valid_metric, save_outputs)

        self.set2idx = registry.get("set2idx")
        self.idx2set = registry.get("idx2set")

    def calculate_metric(self, eval_preds):
        try:
            predictions = eval_preds.predictions
            preds = np.argmax(predictions, axis=1).reshape(-1)
            set_ids = getattr(eval_preds, "label_ids")
        except:
            preds = eval_preds["preds"]
            set_ids = eval_preds["set_ids"]

        labels = np.zeros(preds.shape)
        inputs = self.decoded(eval_preds.inputs) if self.save_outputs else None

        checkpoint_opt_file = registry.get("checkpoint_opt_file")
        save_data = {"preds": preds, "labels": labels, "inputs": inputs, "set_ids": set_ids}
        pickle_write(save_data, checkpoint_opt_file)

        df = pd.DataFrame({
            'labels': labels,
            'predicts': preds,
            'set_ids': set_ids
        })

        grouped = df.groupby('set_ids')

        accuracy_per_set = grouped.apply(lambda x: (x['labels'] == x['predicts']).mean())
        accuracy_per_dict = accuracy_per_set.to_dict()

        bench_results = {self.idx2set[int(key)]: round(accuracy_per_dict[key], 3) for key in accuracy_per_dict}
        mean_acc = sum(list(accuracy_per_dict.values()))/len(list(accuracy_per_dict.values()))
        bench_results["mean accuracy"] = round(mean_acc, 3)
        results = {"result": bench_results}

        self.logger.debug(results)

        return results

    @property
    def metric_name(self):
        return "mean accuracy"
