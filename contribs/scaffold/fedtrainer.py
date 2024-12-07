import copy
import torch
from transformers import TrainerCallback
from peft import get_peft_model_state_dict, set_peft_model_state_dict

from utils.register import registry
from commus.message import Message
from utils.general import cosine_learning_rate, LoadBalanceSampling
from trainers.FedBaseTrainer import BaseTrainer


# This file was originally created by Ye R, Wang W, Chai J, et al. Openfedllm,
# and modified by zhuo on 2024/11/02.
def get_auxiliary_dict(fed_args, global_parameters):

    global_auxiliary = {}
    for key in global_parameters.keys():
        global_auxiliary[key] = torch.zeros_like(global_parameters[key])
    auxiliary_model_list = [copy.deepcopy(global_auxiliary) for _ in range(fed_args.clients_num)]
    auxiliary_delta_dict = [copy.deepcopy(global_auxiliary) for _ in range(fed_args.clients_num)]

    return global_auxiliary, auxiliary_model_list, auxiliary_delta_dict


class scaffold_callback(TrainerCallback):
    def __init__(self, correction, model):
        super(scaffold_callback, self).__init__()
        self.correction = correction
        self.model = model

    def on_step_end(self, args, state, control, **kwargs):
        model_para = copy.deepcopy(get_peft_model_state_dict(self.model))
        for name in model_para.keys():
            self.correction[name] = self.correction[name].to(model_para[name].device)
            model_para[name] -= args.learning_rate * self.correction[name]
        set_peft_model_state_dict(self.model, model_para)


@registry.register_fedtrainer("scaffold")
class ScaffoldTrainer(BaseTrainer):
    def __init__(self, *args):
        super().__init__(*args)
        self._before_training()

        self.global_auxiliary, self.auxiliary_model_list, self.auxiliary_delta_dict = \
            get_auxiliary_dict(self.F, self.model_parameters)

        self.F.weight_type = "num"

    def client_run(self):
        self.client_join()
        while True:
            msg = self.comm_manager.receive()
            if msg.message_type == 101:
                # quit federated learning
                self.on_client_end()
                break
            elif msg.message_type == 200:
                model_parameters = msg.content['model']['update']
                client_ids = msg.content['client_ids'][int(self.F.client_name)]
                auxiliary_model_list = msg.content['model']['auxiliary_model_list'][int(self.F.client_name)]
                global_auxiliary = msg.content['model']['global_auxiliary']
                self.round = msg.content['round']
                self.client_process(client_ids, model_parameters, auxiliary_model_list, global_auxiliary)

    def client_process(self, client_ids, model_parameters, auxiliary_model_list, global_auxiliary):
        param_list, loss_list = {}, {}
        auxiliary_delta_dict_list = {}

        for idx in client_ids:
            train_loss, auxiliary_model, auxiliary_delta_dict = self.client_update(
                idx=idx,
                model_parameters=model_parameters,
                auxiliary_model_list=auxiliary_model_list,
                global_auxiliary=global_auxiliary
            )
            updated_model_parameters = self.serialize_model_parameters()
            param_list[idx] = updated_model_parameters
            loss_list[idx] = train_loss
            auxiliary_delta_dict_list[idx] = auxiliary_delta_dict
            auxiliary_model_list[idx] = auxiliary_model

        model_part = {"update": param_list, 'auxiliary_model_list': auxiliary_model_list,
                      'auxiliary_delta_dict_list': auxiliary_delta_dict_list}
        self.comm_manager.send(
            Message(
                message_type=200,
                sender=self.F.client_name,
                receiver=[self.F.server_ip],
                content={
                    'model': model_part,
                    'loss': loss_list,
                }
            )
        )

    def client_update(self, idx, model_parameters, auxiliary_model_list, global_auxiliary):
        self.logger.debug(f"\n{'=' * 37}\n>>> Subserver={self.F.client_name}_"
                          f"Client={idx}_Round={self.round + 1} <<<\n{'=' * 37}")

        self.deserialize_model(model_parameters)
        train_dataset, eval_dataset = self.get_dataset(idx)

        # manually schedule the learning rate
        self.T.learning_rate = cosine_learning_rate(
            self.round, self.F.rounds, self.eval_args.learning_rate, 1e-6)

        if self.T.max_steps == -1:
            total_bs = registry.get("total_bs")
            max_steps = self.data.train_examples_num_dict[idx] // total_bs
            registry.register("max_steps", max_steps)

        # Initialize local Trainer
        train_op = registry.get_loctrainer(self.T.local_trainer_name)(
            model=self.model,
            args=self.T,
            train_dataset=train_dataset,
            tokenizer=self.data.tokenizer,
            data_collator=self.data.coll_fn(self.model),
            compute_metrics=self.metric.calculate_metric,
            global_parameters=model_parameters,
            local_auxiliary=auxiliary_model_list[idx],
            global_auxiliary=global_auxiliary
        )
        train_op.add_callback(scaffold_callback(train_op.correction, self.model))
        train_result = train_op.train()
        auxiliary_model, auxiliary_delta_dict = train_op.get_auxiliary_param()
        del train_op

        train_loss = round(train_result.training_loss, 3)
        self.logger.info(f">>> Subserver={self.F.client_name}_Client={idx}_lr="
                         f"{self.T.learning_rate * 10000:.2f}e-4_Loss={train_loss}")
        return train_loss, auxiliary_model, auxiliary_delta_dict

    def server_process(self):

        while self.round < self.F.rounds:
            self.client_ids = self.selections[self.round]
            self.metric_log["train_logs"].append([0.0 for _ in range(self.F.client_num_in_total)])
            self.logger.critical(f"Round {self.round + 1} start, Selected Clients: {self.client_ids}")
            balance_sampling = LoadBalanceSampling(self.client_ids, self.F.num_sub)

            client_ids = {}
            auxiliary_model_list = {}
            for i in range(self.F.num_sub):
                client_ids[i] = balance_sampling[i]
                auxiliary_model_list[i] = {idx: self.auxiliary_model_list[idx] for idx in client_ids[i]}

            model_part = {"update": copy.deepcopy(self.model_parameters), 'auxiliary_model_list': auxiliary_model_list,
                          'global_auxiliary': copy.deepcopy(self.global_auxiliary)}
            self.comm_manager.send(
                Message(
                    message_type=200,
                    sender="0",
                    receiver=list(self.comm_manager.communicators.keys()),
                    content={
                        'model': model_part,
                        'client_ids': client_ids,
                        'round': self.round,
                    }
                )
            )

            num_sub = 0
            params_list, loss_list = [], []
            while num_sub < self.F.num_sub:
                msg = self.comm_manager.receive()
                if msg.message_type == 200:
                    num_sub += 1
                    for client_id, params in msg.content['model']["update"].items():
                        params_list.append(params)
                        loss_list.append(msg.content['loss'][client_id])
                        self.metric_log["train_logs"][self.round][client_id] = msg.content['loss'][client_id]

                        self.auxiliary_model_list[client_id] = msg.content['model']['auxiliary_model_list'][client_id]
                        self.auxiliary_delta_dict[client_id] = \
                            msg.content['model']['auxiliary_delta_dict_list'][client_id]

            # aggregation
            self.server_update(params_list, loss_list)

    def server_aggregator(self, serialized_params_list, loss_list):
        weights = self.get_agg_weight(loss_list)
        serialized_parameters = self.serialize_model_parameters()

        for key in serialized_parameters.keys():
            serialized_parameters[key] = sum(
                [client_params[key] * weights[client_id] for client_id, client_params in
                 enumerate(serialized_params_list)]
            )

        for key in self.global_auxiliary.keys():
            delta_auxiliary = sum([self.auxiliary_delta_dict[client_id][key] for client_id in self.client_ids]).cpu()
            self.global_auxiliary[key] += delta_auxiliary / len(self.client_ids)

        return serialized_parameters
