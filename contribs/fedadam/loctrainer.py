

from utils.register import registry
from trainers.LocBaseSFT import LocalSFTTrainer
from trainers.LocBaseDPO import LocalDPOTrainer


@registry.register_loctrainer("fedadagrad")
@registry.register_loctrainer("fedyogi_sft")
@registry.register_loctrainer("fedadam_sft")
class FedAdamLocSFTTrainer(LocalSFTTrainer):
    ...


@registry.register_loctrainer("fedadagrad")
@registry.register_loctrainer("fedyogi_dpo")
@registry.register_loctrainer("fedadam_dpo")
class FedAdamLocDPOTrainer(LocalDPOTrainer):
    ...
