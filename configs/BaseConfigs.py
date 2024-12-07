import os
import time
from abc import ABC
from omegaconf import OmegaConf
from transformers import HfArgumentParser

from utils.register import registry
from utils.logger import setup_logger
from utils.general import make_sure_dirs
from configs.peft_configs import get_delta_key
from configs import ModelArguments, DataTrainingArguments, TrainArguments, FederatedTrainingArguments

# os.environ["WANDB_DISABLED"] = "true"


class Config(ABC):
    """
    Base class for configurations.

    Attributes:
        model_config: ModelArguments
        data_config: DataTrainingArguments
        training_config: TrainingArguments
        federated_config: FederatedTrainingArguments
        M: abbreviation for model_config
        D: abbreviation for data_config
        T: abbreviation for training_config
        F: abbreviation for federated_config

    """
    def __init__(self, model_args, data_args, training_args, federated_args):
        self.model_config = model_args
        self.data_config = data_args
        self.training_config = training_args
        self.federated_config = federated_args

    def check_config(self):
        """
        Check the configurations
        """
        self.config_check_federated()
        self.config_check_tuning()
        self.config_check_trainer()

    def config_check_federated(self):
        _ = self.F.client_num_per_round

    def config_check_valid_len(self):
        if self.F.log_valid_len and self.F.log_valid_len != 0:
            self.F.save_valid_len = 0
        elif self.F.save_valid_len == 0:
            assert self.T.do_zst
        if self.T.do_zst:
            self.F.save_valid_len = 0
            self.F.log_valid_len = 0

    def config_check_trainer(self):
        if self.T.local_rank != -1:
            self.T.ddp_find_unused_parameters = False
        if self.T.test_openai:
            self.T.save_outputs = True
        if self.T.save_outputs:
            self.T.include_inputs_for_metrics = True

    def config_check_tuning(self):

        if not self.M.tuning_type or "fine" in self.M.tuning_type:
            delta_config = {"delta_type": "fine-tuning"}
        else:
            delta_config = {"delta_type": self.M.tuning_type}

        registry.register("delta_config", delta_config)

        if self.M.tuning_type == "lora" and self.M.model_type == "baichuan":
            self.M.target_modules = ["W_pack"]

        for config in [self.T, self.M, self.F, self.D]:
            for key, value in delta_config.items():
                if getattr(config, key, None) is not None:
                    setattr(config, key, value)
                    # registry.debug(f"{key}={value}")
        self.T.tuning_type = delta_config["delta_type"]
        self.M.tuning_type = delta_config["delta_type"]

    @property
    def M(self):
        return self.model_config

    @property
    def D(self):
        return self.data_config

    @property
    def T(self):
        return self.training_config

    @property
    def F(self):
        return self.federated_config

    @property
    def is_fl(self):
        if "cen" in self.F.fl_algorithm:
            return False
        return True


def debug_mode(config):
    """
    Switch the configurations to debug mode

    Args:
        config: Config object

    Returns:
        Config object, updated with debug mode
    """
    if not config.D.debug_mode:
        return config

    if config.is_fl:
        config.F.rounds = 2
        if config.F.log_valid_len:
            config.F.log_valid_len = 1
        config.T.num_train_epochs = 1
    else:
        config.T.num_train_epochs = 2

    config.D.max_train_samples = (
        config.D.max_train_samples if config.D.max_train_samples is not None else 16
    )

    config.D.max_eval_samples = (
        config.D.max_eval_samples if config.D.max_eval_samples is not None else 32
    )

    return config


def build_metric_line(config, times):
    if config.is_fl:
        fed_line = f"cli{config.F.clients_num}_sap{int(config.F.clients_num*config.F.sample)}" \
                   f"_alp{config.F.alpha}_rd{config.F.rounds}"
    else:
        fed_line = ""

    grid_info = "ft"
    if config.M.tuning_type:
        key_name, key_abb = get_delta_key(config.M.tuning_type)
        grid_info = "".join([key_abb, str(getattr(config.M, key_name, ""))])
    elif config.T.do_zst:
        grid_info = "zs"
    registry.register("grid_info", grid_info)

    total_bs = config.T.train_batch_size * config.T.gradient_accumulation_steps * config.T.world_size
    registry.register("total_bs", total_bs)

    metric_line = f"{times}_{config.M.model_type}_{config.M.tuning_type}_" \
                  f"lr{config.T.learning_rate}_epo{config.T.num_train_epochs}_bs{total_bs}_" \
                  f"{fed_line}_{grid_info}_"
    if getattr(config.data_config, "prompt_id", None) is not None:
        prompt_id = config.data_config.prompt_id
        if prompt_id == -1:
            metric_line += f"used_seed={config.data_config.used_seed}_"
        else:
            metric_line += f"prompt_id={prompt_id}_"
    if config.training_config.seed != 42:
        metric_line += f"seed{config.training_config.seed}_"

    registry.register("metric_line", metric_line)

    return metric_line


def amend_config(model_args, data_args, training_args, federated_args):
    """
    Merge the parsed arguments into one config object and amend the configurations
    The config object is registered in the registry for global access

    Returns:
        config: Config object
    """
    config = Config(model_args, data_args, training_args, federated_args)

    # load customer config
    # NOTE: hyper_parameters in config.yaml can overwrite --arg
    root_folder = registry.get("root_folder")
    if config.T.config_path:
        cust_config_path = os.path.join(root_folder, config.T.config_path)
        cust_config = OmegaConf.load(cust_config_path)

        not_overwrite_args = config.T.not_overwrite_args.split(",") if config.T.not_overwrite_args else []
        for key, values in cust_config.items():
            if values:
                args = getattr(config, key)
                for k, v in values.items():
                    if k in not_overwrite_args:
                        # not overwrite --arg
                        continue
                    setattr(args, k, v)

    role = config.F.role
    registry.register(config.F.role, "role")
    # set training path
    config.T.output_dir = os.path.join(config.T.output_dir, config.D.task_name)
    make_sure_dirs(config.T.output_dir, role)
    if not config.T.eval_name:
        config.T.eval_name = config.T.metric_name

    if not config.D.cache_dir:
        cache_dir = os.path.join(config.T.output_dir, "cached_data")
        if config.is_fl:
            config.D.cache_dir = os.path.join(
                cache_dir, f"cached_{config.M.model_type}_{config.F.clients_num}_{config.F.alpha}"
            )
        else:
            config.D.cache_dir = os.path.join(
                cache_dir, f"cached_{config.M.model_type}_centralized"
            )
    make_sure_dirs(config.D.cache_dir, role)

    config.T.save_dir = os.path.join(config.T.output_dir, config.F.fl_algorithm.lower())
    make_sure_dirs(config.T.save_dir, role)

    # set phase
    if config.T.do_zst:
        phase = "zst"
    elif config.T.do_train:
        phase = "train"
    elif config.T.do_eval:
        phase = "eval"
    else:
        phase = "predict"
    registry.register("phase", phase)

    # set metric log path
    config.T.metric_file = os.path.join(config.T.save_dir, f"{config.M.model_type}.eval")

    if config.T.times is None:
        times = time.strftime("%Y%m%d%H%M%S", time.localtime())
        config.T.times = times
    else:
        times = config.T.times
    registry.register("run_time", times)

    # set metric line
    build_metric_line(config, times)

    config.config_check_valid_len()
    if phase == "eval" or phase == "predict":
        if os.path.isdir(config.T.checkpoint_file):
            times = [item for item in config.T.checkpoint_file.split("/") if item][-1].split("_")[0]
            registry.register("run_time", times)
            config.T.times = times
            config.T.checkpoint_opt_file = None
        else:
            assert os.path.isfile(config.T.checkpoint_file)
            config.T.checkpoint_opt_file = config.T.checkpoint_file + ".result.pkl"
    elif phase == "train":
        config.T.checkpoint_dir = os.path.join(config.T.save_dir,
                                               f"{times}_{config.M.model_type}_{config.M.tuning_type}/")
        make_sure_dirs(config.T.checkpoint_dir, role)
        if config.F.save_valid_len:
            # only saving model parameters during training
            config.T.checkpoint_opt_file = None
        elif config.F.log_valid_len:
            # valid model during training, saving best model
            config.T.checkpoint_opt_file = os.path.join(config.T.save_dir,
                                                        f"{times}_{config.M.model_type}_{config.M.tuning_type}/"
                                                        f"best.result.pkl")
        config.T.metric_log_file = os.path.join(config.T.checkpoint_dir,
                                                f"{registry.get('metric_line')[0:-1]}.loss")
        config.T.yaml_file = os.path.join(config.T.checkpoint_dir, f"{times}_{config.M.model_type}.yaml")
    else:
        # zsh
        config.T.checkpoint_opt_file = os.path.join(config.T.save_dir,
                                                    f"{phase}_{config.M.model_type}.pth.result.pkl")
    registry.register("checkpoint_opt_file", config.T.checkpoint_opt_file)
    # disable wandb
    config.T.report_to = []

    # evaluation config
    if config.is_fl and not config.F.pson:
        config.T.evaluation_strategy = "no"
        config.T.predict_with_generate = False
    else:
        if config.F.log_valid_len:
            config.T.evaluation_strategy = "steps"
            config.T.predict_with_generate = True
        else:
            config.T.predict_with_generate = False
    config.T.save_strategy = "no"

    # set generation config
    config.T.generation_max_length = (
        config.T.generation_max_length
        if config.T.generation_max_length is not None
        else 1024
    )
    config.T.generation_num_beams = (
        config.D.num_beams if config.D.num_beams is not None else 1
    )
    config.T.temperature = (
        config.T.temperature if config.T.temperature is not None else 0.8
    )
    config.T.top_p = (
        config.T.top_p if config.T.top_p is not None else 0.8
    )

    # set debug mode
    config = debug_mode(config)
    registry.register("debug", config.data_config.debug_mode)

    if config.T.save_outputs:
        # llm-eval
        eval_key = "test"
    else:
        # benchmark-eval
        eval_key = "train"  # use valid set
    registry.register("eval_key", eval_key)

    # check config
    config.check_config()
    registry.register("config", config)

    # set logger
    setup_logger(config.T)

    # set communication port
    if config.F.role != "server" and (config.F.client_port is None):
        config.F.client_port = config.F.server_port + 1 + config.F.client_name

    return config


def build_config():
    """
    Build configurations required for experiments
    The arguments are automatically parsed from command line and will be merged into one config object
    The config object is registered in the registry for global access

    Returns:
        config: Config object

    """
    # read parameters
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainArguments, FederatedTrainingArguments))
    model_args, data_args, training_args, federated_args = parser.parse_args_into_dataclasses()

    # amend and register configs
    config = amend_config(model_args, data_args, training_args, federated_args)
    metric_line = registry.get("metric_line")

    # logging fl & some path
    logger = registry.get("logger")
    logger.info(f"run phase: {registry.get('phase')}")
    logger.info(f"fl-algorithm: {config.federated_config.fl_algorithm}")
    logger.info(f"output_dir: {config.training_config.output_dir}")
    logger.info(f"cache_dir: {config.data_config.cache_dir}")
    logger.info(f"save_dir: {config.training_config.save_dir}")
    logger.info(f"Total Train Batch: {registry.get('total_bs')}, "
                f"Total Eval Batch: {config.T.eval_batch_size * config.T.world_size}, "
                f"Accumulation Steps: {config.T.gradient_accumulation_steps}, "
                f"GPU Num: {config.T.world_size}, Distributed: {bool(config.T.local_rank != -1)}, "
                f"f16: {config.T.fp16}, bf16: {config.T.bf16}, seed: {config.T.seed}")  # bf16 and fp16
    logger.info(f"TrainBaseInfo: {metric_line}")

    return config
