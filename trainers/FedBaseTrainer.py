import os
import copy
import time
import random
import subprocess

from visualization.visualizer import Visualizer, VisCallback
from commus.message import Message
from commus.communicator import gRPCCommunicationManager
from contribs.centralized.miscs import CenEndEvalStepCallback

from utils.register import registry
from trainers.BaseEngine import BaseEngine
from utils.general import metric_save, model_save
from utils.general import setup_seed, cosine_learning_rate, LoadBalanceSampling, ClientSampling


class BaseTrainer(BaseEngine):
    def __init__(self, *args):
        super().__init__(*args)

    def _build_selections(self):
        self.selections = ClientSampling(
            range(self.F.client_num_in_total), self.F.client_num_per_round,
            self.F.rounds, self.F.client_sample_type)

    def _build_communicators(self):
        if self.is_fl:
            self.logger.info(f"{self.role} building communicators ...")
            if self.role == "server":
                self.logger.debug(f"server build communicator")
                self.comm_manager = gRPCCommunicationManager(
                    ip=self.F.server_ip,
                    port=self.F.server_port,
                    max_connection_num=self.F.num_sub
                )
                self.logger.debug(f"server finish building communicator")
            else:
                time.sleep(random.randint(2, 10))  # wait for server
                self.logger.debug(f"subserver {self.F.client_name} build communicator")
                self.comm_manager = gRPCCommunicationManager(
                    ip=self.F.client_ip,
                    port=self.F.client_port,
                    max_connection_num=1,
                )
                self.comm_manager.add_communicator(
                    communicator_id=self.F.server_ip,
                    communicator_address='{}:{}'.format(self.F.server_ip, self.F.server_port)
                )
        else:
            self.logger.info("local or central training")

    def _before_training(self):
        # set seed
        setup_seed(self.T.seed)

        # build dataset and dataloader
        self._build_data()

        # build federated model
        self._build_model()

        # build metric
        self._build_metric()  # return computer metric

        # global model
        self.best_glo_params = self.serialize_model_parameters()

        # build client selection before building loc trainer
        self._build_selections()

        # build communicators
        self._build_communicators()

        # build visualizer
        self._build_visualizer()

    def run(self):
        """
        Run the trainer according to the phase
        """
        self.logger.critical(f" {self.role.upper()} {self.phase.upper()} START")
        if self.is_fl:
            self.server_run() if self.role == "server" else self.client_run()
        else:
            self.cen_train()
            self.on_cen_end()

    def server_join(self):
        client_num = 0
        while client_num < self.F.num_sub:
            msg = self.comm_manager.receive()
            if msg.message_type == 100:
                client_num += 1
                self.logger.info(f"Subserver {msg.content['client_ip']}:{msg.content['client_port']} join in.")
                self.comm_manager.add_communicator(
                    communicator_id=msg.sender,
                    communicator_address=f"{msg.content['client_ip']}:{msg.content['client_port']}")
                self.logger.info(f"Subserver {msg.sender} joined in ({client_num}/{self.F.num_sub}).")
                self.logger.info(list(self.comm_manager.communicators.keys()))
        self.logger.debug(f"all subserver ({len(self.comm_manager.communicators)}) connect")

    def server_run(self):
        self.server_join()
        self.server_valid()
        self.server_process()
        self.on_server_end()

    def server_process(self):

        while self.round < self.F.rounds:
            self.client_ids = self.selections[self.round]
            self.metric_log["train_logs"].append([0.0 for _ in range(self.F.client_num_in_total)])
            self.logger.critical(f"Round {self.round + 1}/{self.F.rounds} start, Selected Clients: {self.client_ids}")
            balance_sampling = LoadBalanceSampling(self.client_ids, self.F.num_sub)
            client_ids = {}
            for i in range(self.F.num_sub):
                client_ids[i] = balance_sampling[i]

            self.comm_manager.send(
                Message(
                    message_type=200,
                    sender="0",
                    receiver=list(self.comm_manager.communicators.keys()),
                    content={
                        'model': self.model_parameters,
                        'client_ids': client_ids,
                        'round': self.round
                    }
                )
            )

            num_sub = 0
            params_list, loss_list = [], []
            while num_sub < self.F.num_sub:
                msg = self.comm_manager.receive()
                if msg.message_type == 200:
                    num_sub += 1
                    for client_id, params in msg.content['model'].items():
                        params_list.append(params)
                        loss_list.append(msg.content['loss'][client_id])
                        self.metric_log["train_logs"][self.round][client_id] = msg.content['loss'][client_id]

            # aggregation
            self.server_update(params_list, loss_list)

    def server_update(self, param_list, loss_list):
        assert len(param_list) <= self.F.client_num_per_round

        self.round += 1
        should_eval, should_save = False, False
        if self.F.log_valid_len and self.round % self.F.log_valid_len == 0:
            should_eval = True
        if self.F.save_valid_len and self.round % self.F.save_valid_len == 0:
            should_save = True

        self.server_logging(loss_list)

        # Global Aggregation
        serialized_parameters = self.server_aggregator(param_list, loss_list)
        self.deserialize_model(serialized_parameters)

        if should_save and self.phase == "train" and not self.debug:
            checkpoint_file = os.path.join(self.T.checkpoint_dir, f"round-{self.round}")
            self.deserialize_model(serialized_parameters)
            model_save(self.model, self.eval_args, checkpoint_file)
            self.logger.debug(f"Model Saved in: {checkpoint_file}")

        registry.register("round", self.round)
        self.model_parameters = copy.deepcopy(serialized_parameters)

    def server_valid(self):
        if self.T.eval_during_train:
            eval_opts = self.build_eval_cmd()
            eval_opts.extend(["--eval_name", "ayn_local_eval",
                              "--not_overwrite_args", "eval_name",
                              "--checkpoint_file", f"{self.T.checkpoint_dir}"])
            eval_opts = ["python"] + eval_opts
            setattr(self, "eval_opts", subprocess.Popen(eval_opts))
        else:
            setattr(self, "eval_opts", None)

    def server_logging(self, loss_list):
        this_round_loss = sum(loss_list) / len(loss_list)
        self.logger.warning(
            f"FL={self.F.fl_algorithm}_Round={self.round}_ClientNum={len(loss_list)}_"
            f"Loss={this_round_loss:.3f}"
        )
        if self.visualizer is not None:
            self.visualizer.log(
                global_round=self.round,
                data={
                    "train_loss": this_round_loss,
                    # "loss_list": loss_list,
                }
            )

    def server_aggregator(self, serialized_params_list, loss_list):
        """fl algorithm, default fedavg"""
        weights = self.get_agg_weight(loss_list)
        serialized_parameters = self.serialize_model_parameters()

        for key in serialized_parameters.keys():
            serialized_parameters[key] = sum(
                [serialized_params_list[client][key] * weights[client] for client in
                 range(len(serialized_params_list))])

        return serialized_parameters

    def on_server_end(self):
        """Using best parameters for prediction"""
        if not self.debug:
            metric_save(self, self.T, self.logger)

        self.comm_manager.send(
            Message(
                message_type=101,
                sender="0",
                receiver=list(self.comm_manager.communicators.keys()),
                content={
                    '': '',
                }
            )
        )

        if self.eval_opts is not None:
            while self.eval_opts.returncode != 0:
                continue

        self.logger.critical(f"Train done, Please Eval and Test in {self.T.checkpoint_dir}")

    def client_join(self):
        self.comm_manager.send(
            Message(
                message_type=100,
                sender=self.F.client_name,
                receiver=[self.F.server_ip],
                content={
                    'client_ip': self.F.client_ip,
                    'client_port': self.F.client_port
                }
            )
        )
        self.logger.debug(f"Subserver {self.F.client_ip}:{self.F.client_port} join in federated learning")

    def client_run(self):
        # client join in federated learning
        self.client_join()

        while True:
            msg = self.comm_manager.receive()
            if msg.message_type == 101:
                # quit federated learning
                self.on_client_end()
                break
            elif msg.message_type == 200:
                model_parameters = msg.content['model']
                self.round = msg.content['round']
                client_ids = msg.content['client_ids'][int(self.F.client_name)]
                self.client_process(client_ids, model_parameters)
                self.client_valid()
                self.client_logging()

    def client_process(self, client_ids, model_parameters):
        param_list, loss_list = {}, {}
        for idx in client_ids:
            train_loss = self.client_update(
                idx=idx,
                model_parameters=model_parameters
            )
            updated_model_parameters = self.serialize_model_parameters()
            param_list[idx] = updated_model_parameters
            loss_list[idx] = train_loss

        self.comm_manager.send(
            Message(
                message_type=200,
                sender=self.F.client_name,
                receiver=[self.F.server_ip],
                content={
                    'model': param_list,
                    'loss': loss_list
                }
            )
        )

    def client_update(self, idx, model_parameters, *args, **kwargs):
        self.logger.debug(f"\n{'=' * 37}\n>>> Subserver={self.F.client_name}_"
                          f"Client={idx}_Round={self.round + 1} <<<\n{'=' * 37}")

        self.deserialize_model(model_parameters)
        train_dataset, eval_dataset = self.get_dataset(idx)

        # manually schedule the learning rate
        self.T.learning_rate = cosine_learning_rate(
            self.round, self.F.rounds, self.eval_args.learning_rate, 1e-6)

        # Initialize local Trainer
        vis_cb = VisCallback(self.visualizer)
        train_op = registry.get_loctrainer(self.T.local_trainer_name)(
            model=self.model,
            args=self.T,
            train_dataset=train_dataset,
            tokenizer=self.data.tokenizer,
            data_collator=self.data.coll_fn(self.model),
            compute_metrics=self.metric.calculate_metric,
            callbacks=[vis_cb]
            # optimizers
        )
        train_result = train_op.train()
        del train_op

        train_loss = round(train_result.training_loss, 3)
        self.logger.info(f">>> Subserver={self.F.client_name}_Client={idx}_lr="
                         f"{self.T.learning_rate * 10000:.2f}e-4_Loss={train_loss}")
        return train_loss

    def client_valid(self):
        """Valid on Client"""

    def client_logging(self):
        """Logging on Client"""

    def on_client_end(self):
        self.logger.critical(f"Subserver {self.F.client_name} Train done")

    def get_agg_weight(self, loss_list, epsilon=1e-8):
        if self.F.weight_type == "num":
            weights = [self.data.train_examples_num_dict[client_id] for client_id in self.client_ids]
        elif self.F.weight_type == "loss":
            adjusted_losses = [max(loss, epsilon) for loss in loss_list]
            inverse_weights = [1 / loss for loss in adjusted_losses]
            weights = inverse_weights
        else:
            weights = [1.0 for _ in range(len(loss_list))]
        total = sum(weights)
        normalized_weights = [weight / total for weight in weights]
        self.logger.info(f"This round clients' weights: {[round(weight, 3) for weight in normalized_weights]}")
        return normalized_weights

    def cen_train(self, client_id=-1):

        # get local train and eval dataset
        train_dataset, eval_dataset = self.get_dataset(client_id)

        # set some parameters
        total_steps = len(train_dataset) / registry.get("total_bs") * self.T.num_train_epochs
        if self.F.log_valid_len:
            self.T.greater_is_better = False if self.T.is_decreased_valid_metric else True
            self.T.eval_steps = max(int(total_steps / self.F.log_valid_len), 1)

        if self.F.save_valid_len:
            self.T.save_strategy = "steps"
            self.T.save_steps = max(int(total_steps / self.T.num_train_epochs), 1)
            self.T.save_total_limit = int(self.T.num_train_epochs)
            self.T.evaluation_strategy = 'no'

        # Initialize Centralized Trainer
        train_op = registry.get_loctrainer(self.T.local_trainer_name)(
            model=self.model,
            args=self.T,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.data.tokenizer,
            data_collator=self.data.coll_fn(self.model),
            compute_metrics=self.metric.calculate_metric
        )
        train_op.add_callback(CenEndEvalStepCallback(
            train_op, metric_name=self.metric_name)
        )
        train_op.train()

    def on_cen_end(self):
        """Using best parameters for prediction"""
        if not self.debug:
            metric_save(self, self.T, self.logger)

        self.logger.critical(f"Train done, Please Eval and Test in {self.T.checkpoint_dir}")
