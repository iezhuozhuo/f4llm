from copy import deepcopy

import torch
from utils.register import registry
from trainers.FedBaseTrainer import BaseTrainer

# This file was originally created by Ye R, Wang W, Chai J, et al. Openfedllm,
# and modified by zhuo on 2024/11/02.
def get_proxy_dict(fed_alg, global_dict, fedopt_tau=1):
    opt_proxy_dict = None
    proxy_dict = None
    if fed_alg in ['fedadagrad', 'fedyogi', 'fedadam']:
        proxy_dict, opt_proxy_dict = {}, {}
        for key in global_dict.keys():
            proxy_dict[key] = torch.zeros_like(global_dict[key])
            opt_proxy_dict[key] = torch.ones_like(global_dict[key]) * fedopt_tau**2
    elif fed_alg == 'fedavgm':
        proxy_dict = {}
        for key in global_dict.keys():
            proxy_dict[key] = torch.zeros_like(global_dict[key])
    return proxy_dict, opt_proxy_dict


@registry.register_fedtrainer("fedadagrad")
@registry.register_fedtrainer("fedyogi")
@registry.register_fedtrainer("fedadam")
class FedAdamTrainer(BaseTrainer):
    def __init__(self, *args):
        super().__init__(*args)
        self._before_training()

        self.proxy_dict, self.opt_proxy_dict = get_proxy_dict(
            self.F.fl_algorithm, self.model_parameters, fedopt_tau=self.F.fedopt_tau)

    def server_aggregator(self, serialized_params_list, loss_list):
        serialized_parameters = self.serialize_model_parameters()

        fedopt_beta1 = getattr(self.F, "fedopt_beta1", 0.9)
        fedopt_beta2 = getattr(self.F, "fedopt_beta2", 0.99)
        fedopt_eta = getattr(self.F, "fedopt_eta", 1e-3)
        fedopt_tau = getattr(self.F, "fedopt_tau", 1e-3)

        if self.F.fl_algorithm == 'fedyogi':
            for key, param in self.opt_proxy_dict.items():
                delta_w = sum(
                    [(client_params[key] - serialized_parameters[key]) for client_params in
                     serialized_params_list]) / len(serialized_params_list)
                self.proxy_dict[key] = fedopt_beta1 * self.proxy_dict[key] + (
                            1 - fedopt_beta1) * delta_w if (self.round-1) > 0 else delta_w
                delta_square = torch.square(self.proxy_dict[key])
                self.opt_proxy_dict[key] = param - (1 - fedopt_beta2) * delta_square * torch.sign(
                    param - delta_square)
                serialized_parameters[key] += fedopt_eta * torch.div(self.proxy_dict[key], torch.sqrt(
                    self.opt_proxy_dict[key]) + fedopt_tau)

        elif self.F.fl_algorithm == 'fedadagrad':
            for key, param in self.opt_proxy_dict.items():
                delta_w = sum(
                    [(client_params[key] - serialized_parameters[key]) for client_params in
                     serialized_params_list]) / len(serialized_params_list)
                # In paper 'adaptive federated optimization', momentum is not used
                self.proxy_dict[key] = delta_w
                self.opt_proxy_dict[key] = param + torch.square(self.proxy_dict[key])
                serialized_parameters[key] += fedopt_eta * torch.div(self.proxy_dict[key], torch.sqrt(
                    self.opt_proxy_dict[key]) + fedopt_tau)

        elif self.F.fl_algorithm == 'fedadam':
            for key, param in self.opt_proxy_dict.items():
                delta_w = sum(
                    [(client_params[key].cpu() - serialized_parameters[key]) for client_params in serialized_params_list]) / len(
                    serialized_params_list)

                self.proxy_dict[key] = fedopt_beta1 * self.proxy_dict[key] + \
                                       (1 - fedopt_beta1) * delta_w if (self.round-1) > 0 else delta_w
                self.opt_proxy_dict[key] = fedopt_beta2 * param + (1 - fedopt_beta2) * torch.square(self.proxy_dict[key])
                serialized_parameters[key] += fedopt_eta * torch.div(
                    self.proxy_dict[key], torch.sqrt(self.opt_proxy_dict[key]) + fedopt_tau)

        return serialized_parameters
