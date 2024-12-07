""" general functions for FNLP """

import os
import json
import pickle
import math
import random
import shutil
import importlib
import multiprocessing
from glob import glob
from collections import Counter

import torch
import psutil
import numpy as np
from utils.register import registry
from utils.constants import petuning_type
from peft import get_peft_model_state_dict
from transformers import Trainer


"""
This module, `general.py`, encompasses a wide range of utility functions designed to support various aspects of the FNLP (Federated Natural Language Processing) project. It includes functionalities for file operations (reading and writing different formats), directory management, system resource queries (CPU and memory usage), data sampling, learning rate scheduling, sequence padding, model parameter management, random seed setup, and import management. These utilities facilitate the handling of data, ensure reproducibility, optimize resource usage, and dynamically load project components, thereby supporting the project's infrastructure and operational needs.

Key Functions:
- File operations: Reading and writing for pickle, JSON, and plain text files.
- Directory management: Creating and removing directories and files as needed.
- System resource queries: Fetching the number of CPUs and current memory usage.
- Data sampling: Implementing load balance sampling for distributed processing.
- Learning rate scheduling: Calculating learning rates based on a cosine schedule.
- Sequence padding: Custom padding for sequences to a uniform length.
- Model parameter management: Fetching parameters for PEFT (Parameter Efficient Fine-Tuning) models and checking if a model is using PEFT.
- Random seed setup: Ensuring reproducibility across runs by setting random seeds.
- Import management: Dynamically loading project components to facilitate easy extension and customization of the FNLP framework.

This module plays a crucial role in the FNLP project by providing essential utilities that enhance efficiency, maintainability, and scalability of the project's codebase.
"""


def pickle_read(path, read_format="rb"):
    """
    Read pickle file from path and return the object.

    Args:
        path: The path of the pickle file.
        read_format: The read format of the file. Default is "rb".

    Returns:
        The object read from the pickle file.

    """
    with open(path, read_format) as file:
        obj = pickle.load(file)
    return obj


def pickle_write(obj, path, write_format="wb"):
    """
    Write the object to a pickle file at the specified path.

    Args:
        obj: The object to be written.
        path: The path of the pickle file.
        write_format: The write format of the file. Default is "wb".

    Returns:
        None

    """
    with open(path, write_format) as file:
        pickle.dump(obj, file)


def read_json(path_file):
    """
    Read json file from path and return the object.

    Args:
        path_file: The path of the json file.

    Returns:
        The object read from the json file.

    """
    outputs = []
    with open(path_file, "r") as file:
        if path_file.endswith("jsonl"):
            for line in file:
                outputs.append(json.loads(line))
        else:
            outputs = json.load(file)
    return outputs


def write_json(obj, path_file):
    """
    Write the object to a json file at the specified path.

    Args:
        obj: The object to be written.
        path_file: The path of the json file.

    Returns:
        None

    """
    with open(path_file, "w") as file:
        if path_file.endswith("jsonl"):
            for line in obj:
                json.dump(line, file)
                file.write('\n')
        else:
            json.dump(obj, file)


def file_write(line, path, mode):
    """
    Write line to file.

    Args:
        line: The line to be written.
        path: The path of the file.
        mode: The mode of the file.

    Returns:
        None

    """
    with open(path, mode) as file:
        file.write(line + "\n")


def make_sure_dirs(path, role="server"):
    """
    Create dir if not exists and return the path.

    Args:
        path (str): The path to be created.
        role (str): The role of the directory. Default is "server".

    Returns:
        The path of the created directory.

    """
    if role == "client":
        return
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    return path


def rm_dirs(path: str):
    """
    remove file existing check.

    Args:
        path (str): The path to be removed.

    Returns:
        None

    """
    if os.path.exists(path):
        shutil.rmtree(path)


def rm_file(file_path: str):
    """
    remove file existing check.

    Args:
        file_path: The path of the file to be removed.

    Returns:
        None

    """
    if os.path.isfile(file_path):
        os.unlink(file_path)


def get_cpus():
    """
    return total num of cpus in current machine.

    Returns:
        The total number of CPUs in the current machine.

    """
    try:
        with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as f:
            cfs_quota_us = int(f.readline())
        with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as f:
            cfs_period_us = int(f.readline())
        if cfs_quota_us > 0 and cfs_period_us > 0:
            return int(math.ceil(cfs_quota_us / cfs_period_us))
    except Exception:
        pass
    return multiprocessing.cpu_count()


def get_memory_usage():
    """
    return total memory been used (GB).

    Returns:
        The total memory used in GB.
    """
    pid = os.getpid()
    py = psutil.Process(pid)
    memory_use = py.memory_info()[0] / 2.0 ** 30
    return memory_use


def ClientSampling(client_list, num_per_round, rounds, sample_type="random"):
    samples = []

    if sample_type == "random":
        for i in range(rounds):
            samples.append(random.sample(client_list, num_per_round))

    elif sample_type == "coverage":
        N = len(client_list)
        if rounds * num_per_round == N:
            random.shuffle(client_list)
            return [client_list[i * num_per_round:(i + 1) * num_per_round] for i in range(rounds)]
        elif rounds * num_per_round < N:
            samples = []
            used_elements = Counter()
            for i in range(rounds):
                if i == 0:
                    sampled = random.sample(client_list, num_per_round)
                else:
                    available = [x for x in client_list if used_elements[x] < min(used_elements.values())]
                    sampled = random.sample(available, num_per_round)
                samples.append(sampled)
                used_elements.update(sampled)
            return samples
        else:
            full_sets, remaining_samples = divmod(N, num_per_round)
            samples = []
            random.shuffle(client_list)

            for i in range(full_sets):
                samples.append(client_list[i * num_per_round:(i + 1) * num_per_round])

            used_elements = Counter()
            for sample in samples:
                used_elements.update(sample)

            for _ in range(rounds - full_sets):
                available = [x for x in client_list if used_elements[x] == min(used_elements.values())]
                sampled = random.sample(available, num_per_round)
                samples.append(sampled)
                used_elements.update(sampled)
    else:
        raise ValueError(f"client sampling supports [random, coverage], but find {sample_type}")

    return samples


def LoadBalanceSampling(target, split_size):
    """
    Load balance sampling.

    Args:
        target: The target to be sampled.
        split_size: The size of the split.

    Returns:
        The sampled target.

    """
    chunk_size = int(len(target) // split_size)
    result = [target[x:x + chunk_size] for x in range(0, len(target), chunk_size)]

    if len(result) == split_size + 1:
        for i, j in enumerate(result[-1]):
            idx = i % split_size
            result[idx].append(j)
        return result[0:-1]
    elif len(result) == split_size:
        return result
    else:
        raise


def cosine_learning_rate(current_round, total_rounds, initial_lr=0.001, min_lr=0):
    """
    Compute the learning rate based on a cosine schedule.

    Args:
        current_round: The current training round (0-indexed).
        total_rounds: The total number of training rounds.
        initial_lr: The initial learning rate, default is 0.001.
        min_lr: The minimum learning rate, default is 0.

    Returns:
        The computed learning rate for the current round.

    """
    # Compute the cosine learning rate
    cosine_lr = min_lr + 0.5 * (initial_lr - min_lr) * (1 + math.cos(math.pi * current_round / total_rounds))
    return cosine_lr


def custom_pad_sequence(tensor_list, padding_value=-100, left_padding=True):
    """
    Custom padding for sequences to a uniform length.

    Args:
        tensor_list: The list of tensors to be padded.
        padding_value: The value to be used for padding. Default is -100.
        left_padding: Whether to pad on the left side. Default is True.

    Returns:
        The padded sequence.

    """
    # find the longest len
    max_length = max(len(t) for t in tensor_list)

    padded_list = []
    for tensor in tensor_list:
        padding_count = max_length - len(tensor)

        if left_padding:
            # left padding
            padded_tensor = torch.cat([torch.full((padding_count,), padding_value), tensor])
        else:
            # right padding
            padded_tensor = torch.cat([tensor, torch.full((padding_count,), padding_value)])
        padded_list.append(padded_tensor)

    padded_sequence = torch.stack(padded_list)

    return padded_sequence


def get_parameter_number(net):
    """
    Get the total number of parameters and the number of trainable parameters in the model network.

    Args:
        net: The model network.

    Returns:
        A dictionary containing the total number of parameters and the number of trainable parameters
        in the model network.

    """
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': round(total_num / 1e6, 4), 'Trainable': round(trainable_num / 1e6, 4)}


def is_petuning(tuning_type):
    """
    Check if the tuning type is PEFT (Parameter Efficient Fine-Tuning).

    Args:
        tuning_type: The tuning type to be checked.

    Returns:
        True if the tuning type is PEFT, False otherwise.

    """
    for name in petuning_type:
        if name in tuning_type:
            return True
    return False


def get_peft_parameters(model, tuning_type):
    """
    Get the PEFT (Parameter Efficient Fine-Tuning) parameters for the model.

    Args:
        model: The model to be fine-tuned.
        tuning_type: The tuning type to be used.

    Returns:
        The PEFT model state dictionary.

    """
    if tuning_type == "adapter":
        peft_model_state_dict = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                peft_model_state_dict[name] = param
    else:
        peft_model_state_dict = get_peft_model_state_dict(model)

    return peft_model_state_dict


def setup_seed(seed: int):
    """
    Setup seed for reproducibility.

    Args:
        seed: The seed value to be set.

    Returns:
        None

    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False  # maybe slower training


def is_best(p_metric, c_metric, low_is_better):
    """
    Check if the current metric is better than the previous metric.

    Args:
        c_metric: The current metric.
        p_metric: The previous metric.
        low_is_better: Whether a lower metric is better.

    Returns:
        True if the current metric is better than the previous metric, False otherwise

    """
    if not low_is_better and p_metric <= c_metric:
        return True
    elif low_is_better and p_metric >= c_metric:
        return True
    return False


def run_process(proc):
    """
    Run the process.

    Args:
        proc: The process to be run.

    Returns:
        None

    """
    os.system(proc)


def end_log(fun):
    """
    End log wrapper.

    Args:
        fun: The function to be wrapped.

    Returns:
        The wrapped function.

    """
    def wapper(handler_or_trainer, training_config, logger):
        """
        Wapper function.

        Args:
            handler_or_trainer: The handler or trainer object.
            training_config: The training configuration.
            logger: The logger object.

        Returns:
            The wrapped function.

        """
        if training_config.local_rank <= 0:
            logger.info(f"see training logs --> {training_config.metric_log_file}")
            logger.info(f"see training results --> {training_config.metric_file}")
            return fun(handler_or_trainer, training_config, logger)

    return wapper


@end_log
def metric_save(trainer, training_config, logger=None):
    """
    Save the metric to the file.

    Args:
        trainer: The trainer object.
        training_config: The training configuration.
        logger: The logger object, default is None.

    Returns:
        None

    """
    pickle_write(trainer.metric_log, training_config.metric_log_file)
    # trainer.metric_line += f"valid_{trainer.metric_name}={trainer.global_valid_best_metric:.3f}_"
    # trainer.metric_line += f"test_{trainer.global_test_best_metric}"
    # file_write(trainer.metric_line, training_config.metric_file, "a+")


def model_save(model, args, checkpoint_file):
    save_op = Trainer(
        model=model,
        args=args
    )
    save_op.save_model(checkpoint_file)


def setup_imports():
    """
    Setup imports for the project.

    Returns:
        None
    """
    from utils.register import registry
    # First, check if imports are already setup
    has_already_setup = registry.get("imports_setup", no_warning=True)
    if has_already_setup:
        return
    # Automatically load all of the modules, so that they register with registry
    root_folder = os.path.dirname(os.path.abspath(__file__))
    project_name = root_folder.split(os.sep)[-2]
    root_folder = os.path.join(root_folder, "..")  # check here
    files = []
    for package_name in ["trainers", "contribs", "models", "datas", "utils", "configs", "evals", "metrics", "visualization"]:
        folder = os.path.join(root_folder, package_name)
        pattern = os.path.join(folder, "**", "*.py")
        files.extend(glob(pattern, recursive=True))

    for f in files:
        f = os.path.realpath(f)
        if f.endswith(".py") and not f.endswith("__init__.py"):
            splits = f.split(os.sep)
            import_prefix_index = 0
            for idx, split in enumerate(splits):
                if split == project_name:
                    import_prefix_index = idx + 1
            file_name = splits[-1]
            module_name = file_name[: file_name.find(".py")]
            module = ".".join(
                splits[import_prefix_index:-1] + [module_name]
            )
            importlib.import_module(module)

    registry.register("root_folder", root_folder)
    registry.register("imports_setup", True)
