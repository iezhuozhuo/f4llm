

from utils.register import registry
from trainers.FedBaseTrainer import BaseTrainer


@registry.register_fedtrainer("centralized")
class CenTrainer(BaseTrainer):
    def __init__(self, *args):
        super().__init__(*args)
        self._before_training()
