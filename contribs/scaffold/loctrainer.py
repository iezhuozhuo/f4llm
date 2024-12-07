
import copy
import torch

from utils.register import registry
from trainers.LocBaseSFT import LocalSFTTrainer
from trainers.LocBaseDPO import LocalDPOTrainer


@registry.register_loctrainer("scaffold_sft")
class ScaffoldSFTLocTrainer(LocalSFTTrainer):
    def __init__(self, global_parameters, local_auxiliary, global_auxiliary, **kwargs):
        super(ScaffoldSFTLocTrainer, self).__init__(**kwargs)

        self.global_state = global_parameters
        self.local_auxiliary = local_auxiliary
        self.global_auxiliary = global_auxiliary
        self.correction = copy.deepcopy(local_auxiliary)

        for name in self.correction.keys():
            self.correction[name] = self.global_auxiliary[name] - self.local_auxiliary[name]

    def get_auxiliary_param(self):
        auxiliary_new_para = copy.deepcopy(self.local_auxiliary)
        auxiliary_delta_para = copy.deepcopy(self.local_auxiliary)
        max_steps = registry.get("max_steps")

        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue
                else:
                    name = name.replace(".default", "")
                    auxiliary_new_para[name] = (self.global_state[name] - param) / (
                                max_steps * self.args.learning_rate) - self.correction[name]
                    auxiliary_delta_para[name] = auxiliary_new_para[name] - self.local_auxiliary[name]

        return auxiliary_new_para, auxiliary_delta_para


@registry.register_loctrainer("scaffold_dpo")
class ScaffoldDPOLocTrainer(LocalDPOTrainer):
    def __init__(self, global_parameters, local_auxiliary, global_auxiliary, **kwargs):
        super(ScaffoldDPOLocTrainer, self).__init__(**kwargs)

        self.global_state = global_parameters
        self.local_auxiliary = local_auxiliary
        self.global_auxiliary = global_auxiliary
        self.correction = copy.deepcopy(local_auxiliary)

        for name in self.correction.keys():
            self.global_auxiliary[name] = self.global_auxiliary[name].to(self.local_auxiliary[name].device)
            self.correction[name] = self.global_auxiliary[name] - self.local_auxiliary[name]

    def get_auxiliary_param(self):
        auxiliary_new_para = copy.deepcopy(self.local_auxiliary)
        auxiliary_delta_para = copy.deepcopy(self.local_auxiliary)
        max_steps = registry.get("max_steps")

        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue
                else:
                    name = name.replace(".default", "")
                    self.global_state[name] = self.global_state[name].to(param.device)
                    self.local_auxiliary[name] = self.local_auxiliary[name].to(param.device)
                    auxiliary_new_para[name] = (self.global_state[name] - param) / (
                                max_steps * self.args.learning_rate) - self.correction[name]
                    auxiliary_delta_para[name] = auxiliary_new_para[name] - self.local_auxiliary[name]

        return auxiliary_new_para, auxiliary_delta_para
