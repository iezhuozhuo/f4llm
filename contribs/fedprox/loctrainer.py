
import torch
from utils.register import registry
from trainers.LocBaseSFT import LocalSFTTrainer
from trainers.LocBaseDPO import LocalDPOTrainer


@registry.register_loctrainer("fedprox_sft")
class FedProxLocSFTTrainer(LocalSFTTrainer):
    def __init__(self, global_state, prox_mu, **kwargs):
        super(FedProxLocSFTTrainer, self).__init__(**kwargs)
        self.global_state = global_state
        self.mu = prox_mu

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):

        return_values = super(FedProxLocSFTTrainer, self).compute_loss(
            model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch)

        if return_outputs:
            loss, outputs = return_values
        else:
            loss = return_values

        # Apply FedProx Loss
        for name, param in model.named_parameters():
            name = name.replace(".default", "")
            name = name.replace("module.", "")
            # only trainable parameters
            if not param.requires_grad:
                continue
            else:
                self.global_state[name] = self.global_state[name].to(param.device)
                loss += self.mu / 2 * torch.norm(param - self.global_state[name]) ** 2

        return (loss, outputs) if return_outputs else loss


@registry.register_loctrainer("fedprox_dpo")
class FedProxLocDPOTrainer(LocalDPOTrainer):
    def __init__(self, global_state, prox_mu, **kwargs):
        super(FedProxLocDPOTrainer, self).__init__(**kwargs)
        self.global_state = global_state
        self.mu = prox_mu

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):

        return_values = super(FedProxLocDPOTrainer, self).compute_loss(
            model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch)

        if return_outputs:
            loss, outputs = return_values
        else:
            loss = return_values

        # Apply FedProx Loss
        for name, param in model.named_parameters():
            name = name.replace(".default", "")
            name = name.replace("module.", "")
            # only trainable parameters
            if not param.requires_grad:
                continue
            else:
                self.global_state[name] = self.global_state[name].to(param.device)
                loss += self.mu / 2 * torch.norm(param - self.global_state[name]) ** 2

        return (loss, outputs) if return_outputs else loss

