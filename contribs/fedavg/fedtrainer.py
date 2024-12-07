

from utils.register import registry
from trainers.FedBaseTrainer import BaseTrainer


@registry.register_fedtrainer("fedavg")
class FedAvgTrainer(BaseTrainer):
    def __init__(self, *args):
        super().__init__(*args)
        self._before_training()
