from typing import Optional
from dataclasses import dataclass, field


@dataclass
class FederatedTrainingArguments:
    fl_algorithm: str = field(
        default="fedavg",
        metadata={"help": "The name of the federated learning algorithm"},
    )
    clients_num: int = field(
        default=10,
        metadata={"help": "The number of participant clients"},
    )
    alpha: Optional[float] = field(
        default=1.0,
        metadata={"help": "Non-IID shift and None denotes IID"},
    )
    partition_method: str = field(
        default=None,
        metadata={"help": "The partition methods"},
    )
    do_mimic: bool = field(
        default=True,
        metadata={"help": "Only process data processing in server if True"}
    )
    rounds: int = field(
        default=10, metadata={"help": "The number of training round"}
    )
    sample: float = field(
        default=1, metadata={"help": "The participant ratio in each training round"}
    )
    client_sample_type: str = field(
        default="random",
        metadata={"help": "How to sample client in each round (random or coverage)."},
    )
    role: str = field(
        default="server",
        metadata={"help": "Important! The role of running scripts"}
    )
    num_sub: int = field(
        default=2,
        metadata={"help": "The number of subserver."}
    )
    client_name: int = field(
        default=0,
        metadata={"help": "Important! The role of running scripts"}
    )
    server_ip: str = field(
        default="127.0.0.0",
        metadata={"help": "Important! The ip address of server."}
    )
    server_port: str = field(
        default="10001",
        metadata={"help": "Important! The commu port of server."}
    )
    client_ip: str = field(
        default="127.0.0.0",
        metadata={"help": "Important! The ip address of server."}
    )
    client_port: str = field(
        default=None,
        metadata={"help": "Important! The commu port of server."}
    )
    pson: bool = field(
        default=False, metadata={"help": "Whether to use personalized test(local) metric"}
    )
    log_valid_len: int = field(
        default=None, metadata={"help": "logging valid(global) metric"}
    )
    save_valid_len: int = field(
        default=None, metadata={"help": "saving valid(global) trainable model"}
    )
    test_rounds: bool = field(
        default=False, metadata={"help": "logging test(global) metric"}
    )
    log_test_len: int = field(
        default=10, metadata={"help": "logging test per communication rounds"}
    )
    weight_type: str = field(
        default=None,
        metadata={"help": "Use for weight aggregation"},
    )
    ldp_delta: float = field(
        default=0.001, metadata={"help": "A hyper-parameter for LDP."}
    )
    ldp_privacy_budget: float = field(
        default=10, metadata={"help": "A hyper-parameter for LDP."}
    )
    use_ldp: bool = field(
        default=False, metadata={"help": "use local differential privacy? default is false."}
    )
    fedopt_tau: Optional[float] = field(default=1e-3,
                                        metadata={"help": "the tau parameter of FedAdagrad, FedYogi and FedAdam"})
    fedopt_eta: Optional[float] = field(default=1e-3, metadata={
        "help": "the global learning rate parameter of FedAdagrad, FedYogi and FedAdam"})
    fedopt_beta1: Optional[float] = field(default=0.9, metadata={"help": "the beta1 parameter of FedYogi and FedAdam"})
    fedopt_beta2: Optional[float] = field(default=0.99, metadata={"help": "the beta2 parameter of FedYogi and FedAdam"})

    def __post_init__(self):
        if self.alpha is None:
            # IID
            self.alpha = "inf"

        if not self.do_mimic:
            print("Please check whether federated device has its own data")

    @property
    def is_fl(self):
        if "cen" in self.fl_algorithm:
            return False
        else:
            return True

    @property
    def clients_id_list(self):
        # if self.is_fl:
        #     return [1]
        # else:
        #     client_id_list = [i for i in range(self.clients_num)]
        #     return client_id_list
        client_id_list = [i for i in range(self.clients_num)]
        return client_id_list

    @property
    def client_num_in_total(self):
        if "cen" in self.fl_algorithm:
            # centralized
            return 1
        else:
            # federated
            return self.clients_num

    @property
    def client_num_per_round(self):
        if "cen" in self.fl_algorithm:
            # centralized
            return 1
        else:
            # federated
            assert 0 < self.sample <= 1
            return int(self.clients_num * self.sample)

    @property
    def partition_name(self):
        if self.partition_method:
            return self.partition_method

        return f"clients={self.clients_num}_alpha={self.alpha}"
