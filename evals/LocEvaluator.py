from abc import ABC

from utils.register import registry
from evals.BaseEvaluator import BaseEvaluator


@registry.register_eval("local")
class LocEvaluator(BaseEvaluator, ABC):
    def __init__(self, *args):
        super().__init__(*args)

    def build_eval_op(self, model=None):
        eval_model = model if model != None else self.model
        eval_op = registry.get_loctrainer(self.T.local_trainer_name)(
            model=eval_model,
            args=self.eval_args,
            tokenizer=self.data.tokenizer,
            data_collator=self.data.coll_fn(eval_model),
            compute_metrics=self.metric.calculate_metric,
        )
        return eval_op
