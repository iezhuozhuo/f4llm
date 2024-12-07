

from utils.register import registry
from trainers.LocBaseSFT import LocalSFTTrainer
from trainers.LocBaseDPO import LocalDPOTrainer
from trainers.LocBaseRM import LocalRMTrainer

from trl import DPOTrainer


@registry.register_loctrainer("fedavg_sft")
class FedAvgSFTLocTrainer(LocalSFTTrainer):
    ...


@registry.register_loctrainer("fedavg_trl_dpo")
class FedAvgDPOLocTrainer(DPOTrainer):
    ...


@registry.register_loctrainer("fedavg_dpo")
class FedAvgDPOLocTrainer(LocalDPOTrainer):
    ...


@registry.register_loctrainer("fedavg_rm")
class FedAvgRMLocTrainer(LocalRMTrainer):
    ...
