
import os
import numpy as np
from copy import deepcopy

import torch
from transformers import TrainerCallback

from utils.register import registry
from utils.general import is_petuning
from utils.serialization import SerializationTool


# build callback for centralized
class CenEndEvalStepCallback(TrainerCallback):
    def __init__(self, trainer, best_valid_metric=0.0, metric_name=None) -> None:
        super().__init__()
        self._trainer = trainer
        self.logger = registry.get("logger")
        self.best_valid_metric = best_valid_metric
        self.metric_name = metric_name
        self.greater_is_better = self._trainer.args.greater_is_better
        self.best = False

    def on_step_end(self, args, state, control, **kwargs):
        if control.should_evaluate:
            control_copy = deepcopy(control)
            eval_result = self._trainer.evaluate(eval_dataset=self._trainer.eval_dataset, metric_key_prefix="eval")
            eval_metrics = eval_result["eval_result"]
            eval_metric = eval_metrics[self.metric_name]

            if not self.greater_is_better and self.best_valid_metric >= eval_metric:
                self.best = True
                self.best_valid_metric = eval_metric
                registry.register("best_glo_params", self.serialize_model_parameters())
                registry.register("best_valid_metric", self.best_valid_metric)
            elif self.greater_is_better and self.best_valid_metric <= eval_metric:
                self.best = True
                self.best_valid_metric = eval_metric
                registry.register("best_glo_params", self.serialize_model_parameters())
                registry.register("best_valid_metric", self.best_valid_metric)
            else:
                self.best = False

            self.logger.debug(f"Centralized Eval, Global Steps: {self._trainer.state.global_step}, "
                              f"Current {self.metric_name}: {eval_metric:.3f}, "
                              f"Best {self.metric_name}: {self.best_valid_metric:.3f}, "
                              f"Best Steps: {self.best}")

            control.should_evaluate = False
            control_copy.should_evaluate = False

        if control.should_save:
            control_copy = deepcopy(control)
            checkpoint_file = os.path.join(self._trainer.args.checkpoint_dir, f"steps={state.global_step}")
            self._trainer.save_model(checkpoint_file)
            # serialized_parameters = self.serialize_model_parameters()
            # torch.save(serialized_parameters, checkpoint_file)

            control.should_save = False
            control_copy.should_save = False

            return control_copy

    def serialize_model_parameters(self):
        if is_petuning(self._trainer.args.tuning_type):
            model_parameters = SerializationTool.serialize_peft_model(
                self._trainer.model, tuning_type=self._trainer.args.tuning_type)
        else:
            model_parameters = SerializationTool.serialize_model(self._trainer.model)
        return model_parameters


def decoded_data(preds, labels, tokenizer):
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    return decoded_preds, decoded_labels
