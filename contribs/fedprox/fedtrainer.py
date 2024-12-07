
from copy import deepcopy

from utils.register import registry
from utils.general import cosine_learning_rate
from trainers.FedBaseTrainer import BaseTrainer


@registry.register_fedtrainer("fedprox")
class FedProxTrainer(BaseTrainer):
    def __init__(self, *args):
        super().__init__(*args)
        self._before_training()

    def client_update(self, idx, model_parameters, *args, **kwargs):
        self.logger.debug(f"\n{'=' * 35}\n>>> Subserver={self.F.client_name}_"
                          f"Client={idx}_Round={self.round + 1} <<<\n{'=' * 35}")

        self.deserialize_model(model_parameters)
        train_dataset, eval_dataset = self.get_dataset(idx)

        # manually schedule the learning rate
        self.T.learning_rate = cosine_learning_rate(
            self.round, self.F.rounds, self.eval_args.learning_rate, 1e-6)

        prox_mu = getattr(self.F, "prox_mu", 0.1)
        # Initialize local Trainer
        train_op = registry.get_loctrainer(self.T.local_trainer_name)(
            model=self.model,
            args=self.T,
            train_dataset=train_dataset,
            tokenizer=self.data.tokenizer,
            data_collator=self.data.coll_fn(self.model),
            compute_metrics=self.metric.calculate_metric,
            global_state=deepcopy(model_parameters),
            prox_mu=prox_mu
            # callbacks=None
            # optimizers
        )
        train_result = train_op.train()
        del train_op

        train_loss = round(train_result.training_loss, 3)
        self.logger.info(f">>> Subserver={self.F.client_name}_Client={idx}_lr="
                         f"{self.T.learning_rate * 10000:.2f}e-4_Loss={train_loss}")
        return train_loss
