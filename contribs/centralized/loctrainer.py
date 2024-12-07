

from utils.register import registry
from trainers.LocBaseSFT import LocalSFTTrainer
from trainers.LocBaseDPO import LocalDPOTrainer


@registry.register_loctrainer("centralized_sft")
class CenSFTTrainer(LocalSFTTrainer):
    ...


@registry.register_loctrainer("centralized_dpo")
class CenDPOTrainer(LocalDPOTrainer):
    ...
