"""Base DataLoader"""

import os
import numpy as np
from abc import ABC

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, DataCollatorForSeq2Seq

from utils.register import registry
from utils.general import pickle_read, pickle_write

"""
This module provides the basic structure and methods for building and managing federated datasets, and the basic data manager for federated learning. 
It defines the abstract class 'FedBaseDataset' for federated datasets and the abstract class 'FedBaseDataManger' for federated data management in NLP tasks.
"""


class FedBaseDataset(Dataset):
    """
    Base class for federated datasets

    Attributes:
        data (list): List of data examples
    """

    def __init__(self, features, **kv):
        self.data = features

        for k, v in kv.items():
            setattr(self, k, v)

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)

    def select(self, max_samples):
        """
        Select a subset of the dataset

        Args:
            max_samples (int): Maximum number of samples to select
        Returns:
            FedBaseDataset: A subset of the dataset

        Examples:
            >>> dataset = FedBaseDataset(features)
            >>> subset = dataset.select(100)
            >>> print(len(subset)) # 100
        """
        # max_samples = min(len(self), max_samples)
        features = self.data[0:max_samples]
        return FedBaseDataset(features)


class FedBaseDataManger(ABC):
    """
    Base class for data management in Federated Learning for NLP tasks

    Attributes:
        model_config (configs.ModelArguments): Model configuration
        data_config (configs.DataArguments): Data configuration
        training_config (configs.TrainingArguments): Training configuration
        federated_config (configs.FederatedTrainingArguments): Federated configuration
        is_fl (bool): Whether the task is federated learning
        partition_name (str): Name of the partition
        clients_list (list): List of client IDs
        logger (Logger): Logger object
        ignore_index (int): Index to ignore in the loss function
        model_max_length (int): Maximum length of the model input
    """

    def __init__(self):

        config = registry.get("config")
        self.model_config = config.model_config
        self.data_config = config.data_config
        self.training_config = config.training_config
        self.federated_config = config.federated_config

        self.is_fl = config.is_fl
        self.partition_name = self.federated_config.partition_name
        self.clients_list = self.federated_config.clients_id_list
        self.logger = registry.get("logger")

        self.ignore_index = self.data_config.ignore_index
        self.model_max_length = self.data_config.model_max_length

        self._load_attributes()
        self._build_tokenizer()
        self._build_registry()

        self.train_dataset_dict = {}
        self.valid_dataset_dict = {}
        self.test_dataset_dict = {}
        self.train_examples_num_dict = {}

    def load_data(self):
        """
        Load the dataset and build the federated dataset
        """

        train_dataset_dict, valid_dataset_dict, test_dataset_dict = {}, {}, {}
        train_features_all, valid_features_all, test_features_all = [], [], []

        train_features_dict, valid_features_dict, test_features_dict, train_examples_num_dict, \
        valid_examples_num_dict, test_examples_num_dict, train_num, valid_num, test_num = self._load_cached_data()

        for idx in range(self.attribute["clients_num"]):
            train_dataset_dict[idx] = self.build_dataset(train_features_dict[idx])
            valid_dataset_dict[idx] = self.build_dataset(valid_features_dict[idx]) \
                if len(valid_features_dict[idx]) != 0 else None
            test_dataset_dict[idx] = self.build_dataset(test_features_dict[idx]) \
                if len(test_features_dict[idx]) != 0 else None

            train_features_all += list(train_features_dict[idx])
            valid_features_all += list(valid_features_dict[idx])
            test_features_all += list(test_features_dict[idx])

        train_dataset_dict[-1] = self.build_dataset(train_features_all)
        valid_dataset_dict[-1] = self.build_dataset(valid_features_all) if valid_features_all else None
        test_dataset_dict[-1] = self.build_dataset(test_features_all) if test_dataset_dict else None

        self.train_dataset_dict = train_dataset_dict
        self.valid_dataset_dict = valid_dataset_dict
        self.test_dataset_dict = test_dataset_dict

        self.train_examples_num_dict = train_examples_num_dict
        self.logger.info(f"Train num: {self.train_num}, "
                         f"Valid num: {self.valid_num}, "
                         f"Test num: {self.test_num}")

    def _load_cached_data(self):
        """
        Load data from the file system (or cache if available)
        """
        with self.training_config.main_process_first(desc="Dataset pre-processing"):
            if not self.data_config.overwrite_cache and os.path.isfile(self.cached_data_file):
                self.logger.info(f"loading cached data from {self.cached_data_file}")
                train_features_dict, valid_features_dict, test_features_dict, train_examples_num_dict, \
                valid_examples_num_dict, test_examples_num_dict, self.train_num, self.valid_num, self.test_num \
                    = pickle_read(self.cached_data_file)
            else:
                self.logger.info(f"generating cached data ...")
                train_features_dict, valid_features_dict, test_features_dict, train_examples_num_dict, \
                valid_examples_num_dict, test_examples_num_dict, self.train_num, self.valid_num, self.test_num \
                    = self._convert_examples_to_features()

        return train_features_dict, valid_features_dict, test_features_dict, train_examples_num_dict, \
               valid_examples_num_dict, test_examples_num_dict, self.train_num, self.valid_num, self.test_num

    def _convert_examples_to_features(self):
        """
        Read the raw data from file and convert it into features, then cache the features into a file

        Returns:
            tuple: A tuple containing the following elements:
                - train_features_dict (dict): A dictionary containing the training features for each client
                - valid_features_dict (dict): A dictionary containing the validation features for each client
                - test_features_dict (dict): A dictionary containing the test features for each client
                - train_examples_num_dict (dict): A dictionary containing the number of training examples for each client
                - valid_examples_num_dict (dict): A dictionary containing the number of validation examples for each client
                - test_examples_num_dict (dict): A dictionary containing the number of test examples for each client
                - train_num (int): The total number of training examples
                - valid_num (int): The total number of validation examples
                - test_num (int): The total number of test examples
        """
        raw_data = pickle_read(self.data_config.raw_dataset_path)
        partition_data = pickle_read(self.data_config.partition_dataset_path)

        if self.partition_name in partition_data:
            partition_data = partition_data[self.partition_name]

        train_features_dict, valid_features_dict, test_features_dict = \
            {}, {}, {}
        train_examples_num_dict, valid_examples_num_dict, test_examples_num_dict = \
            {}, {}, {}

        n_clients = self.attribute["clients_num"]
        if n_clients != self.federated_config.clients_num:
            raise ValueError(f"partition data have {n_clients} clients "
                             f"that mismatches your input {self.federated_config.clients_num} clients")

        self.logger.info("convert train examples into features ...")
        train_features_all = np.array(self.process_examples(raw_data["train"], "train"))

        self.logger.info("convert valid examples into features ...")
        if "valid" not in raw_data:
            valid_features_all = []
        else:
            valid_features_all = np.array(self.process_examples(raw_data["valid"], "valid"))

        self.logger.info("convert test examples into features ...")
        if "test" not in raw_data:
            test_features_all = []
        else:
            test_features_all = np.array(self.process_examples(raw_data["test"], "test"))

        self.logger.info("build clients train & valid features ...")
        for idx in range(n_clients):
            client_train_list = partition_data["train"][idx]
            train_examples_num_dict[idx] = len(client_train_list)
            train_features_dict[idx] = train_features_all[client_train_list]

            if "valid" in partition_data:
                client_valid_list = partition_data["valid"][idx]
                valid_examples_num_dict[idx] = len(client_valid_list)
                valid_features_dict[idx] = valid_features_all[client_valid_list]
            else:
                valid_examples_num_dict[idx], valid_features_dict[idx] = 0, []

            if "test" in partition_data:
                client_test_list = partition_data["test"][idx]
                test_examples_num_dict[idx] = len(client_test_list)
                test_features_dict[idx] = test_features_all[client_test_list]
            else:
                test_examples_num_dict[idx], test_features_dict[idx] = 0, []

        self.train_num, self.valid_num, self.test_num = \
            len(train_features_all), len(valid_features_all), len(test_features_all)

        federated_data = (
            train_features_dict, valid_features_dict, test_features_dict,
            train_examples_num_dict, valid_examples_num_dict, test_examples_num_dict,
            self.train_num, self.valid_num, self.test_num
        )

        pickle_write(federated_data, self.cached_data_file)
        self.logger.info(f"processed features saved in {self.cached_data_file}")

        return federated_data

    def process_examples(self, examples, mode="train"):
        """
        Process the examples (dict of texts) for NLP tasks.
        The input(text_a) and label(label) in each example will both be tokenized, and will be concatenated to form
        the input_ids and label_ids. The label_ids will have the input part masked with the ignore_index.
        Both tokenized sequences are padded to the model_max_length.

        Args:
            examples (list[dict]): List of examples, each of which is a dictionary containing the text_a and label
            mode (str): Mode of the examples (train, valid, test)

        Returns:
            list[dict]: List of processed examples, each of which is a dictionary containing the input_ids, labels,
                        attention_mask, and idx
        """
        instances = []

        for idx, example in enumerate(examples):
            text, label = example["text_a"], example["label"]

            # build input with instructions
            input_text = self.build_inputs(self.prompt_text, text)
            src_ids = self.tokenizer.encode(input_text, add_special_tokens=False)
            # if len(src_ids) > self.data_config.max_src_length:
            #     src_ids = src_ids[:self.data_config.max_src_length]

            tgt_ids = self.tokenizer.encode(label, add_special_tokens=False)
            # if len(tgt_ids) > self.max_tgt_len:
            #     tgt_ids = tgt_ids[:self.max_tgt_len]

            context_length = len(src_ids)
            if mode == "train":
                input_ids = src_ids + tgt_ids + [self.tokenizer.eos_token_id]
                label_ids = [self.tokenizer.pad_token_id] * context_length + tgt_ids + [self.tokenizer.eos_token_id]
            else:
                input_ids = src_ids
                label_ids = tgt_ids

            input_ids = input_ids[: self.model_max_length]
            label_ids = label_ids[: self.model_max_length]

            # training/evaluate/predict with left padding -->  bad performance for baichuan2
            # pad_len = self.model_max_length - len(input_ids)
            # if mode == "train":
            #     input_ids = [self.tokenizer.pad_token_id] * pad_len + input_ids
            #     label_ids = [self.tokenizer.pad_token_id] * pad_len + label_ids
            # else:
            #     input_ids = [self.tokenizer.pad_token_id] * pad_len + input_ids
            # label_ids = [(l if l != self.tokenizer.pad_token_id else self.ignore_index) for l in label_ids]

            # training with right padding & evaluate/predict with left padding
            pad_len = self.model_max_length - len(input_ids)
            if mode == "train":
                input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_len
                label_ids = label_ids + [self.tokenizer.pad_token_id] * pad_len
            else:
                input_ids = [self.tokenizer.pad_token_id] * pad_len + input_ids
            label_ids = [(l if l != self.tokenizer.pad_token_id else self.ignore_index) for l in label_ids]

            input_ids = torch.LongTensor(input_ids)
            label_ids = torch.LongTensor(label_ids)
            attention_mask = input_ids.ne(self.tokenizer.pad_token_id).long()

            instances.append(
                {"input_ids": input_ids, "labels": label_ids,
                 "attention_mask": attention_mask, "idx": f"{mode}-{idx}"}
            )

        return instances

    def coll_fn(self, model):
        """
        Build the data collection function.
        Returns:
            DataCollatorForSeq2Seq: Data collator for the model
        """
        data_collator = DataCollatorForSeq2Seq(
            self.tokenizer,
            model=model,
            label_pad_token_id=self.ignore_index,
            pad_to_multiple_of=None,
            padding=False
        )
        return data_collator

    def _load_attributes(self):
        """
        Load the attributes of the federated dataset, such as the number of clients
        """
        partition_data = pickle_read(self.data_config.partition_dataset_path)
        if self.partition_name in partition_data:
            partition_data = partition_data[self.partition_name]
        self.attribute = partition_data["attribute"]
        self.clients_num = self.attribute["clients_num"]

    def _build_registry(self):
        """
        Register the attributes of the federated dataset, used for xlms
        """
        if 'lang_map' in self.attribute:
            registry.register("eval_batch", self.training_config.per_device_eval_batch_size)
            registry.register("lang_map", self.attribute["lang_map"])

        if 'subset_map' in self.attribute:
            # used for reward_benchmark
            registry.register("subset_map", self.attribute["subset_map"])

    def build_dataset(self, features):
        """
        Build the federated dataset

        Args:
            features (list): List of examples

        Returns:
            FedBaseDataset: The federated dataset
        """
        dataset = FedBaseDataset(features)
        return dataset

    def _build_tokenizer(self):
        """
        Initialize self.tokenizer and set up its special tokens
        """
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_config.model_name_or_path,
            trust_remote_code=True,
            use_fast=False,
            model_max_length=self.model_max_length
        )
        # if self.model_config.model_type in ["llama2-base", "tinyllama"]:
        if "llama" in self.model_config.model_type:
            self.tokenizer.pad_token = self.tokenizer.unk_token
            # tokenizer.pad_token = tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.unk_token_id
        elif self.model_config.model_type in ["qwen"]:
            self.tokenizer.eos_token = self.tokenizer.decode(self.tokenizer.eod_id)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eod_id

    def build_inputs(self, prompt_text, text):
        inputs_text = prompt_text.format(text)
        return inputs_text

    @property
    def cached_data_file(self):
        """
        Get the path to the cached data file

        Returns:
            str: Path to the cached data file
        """
        cached_file_name = f"models={self.model_config.model_type}_" \
                           f"seq={self.data_config.model_max_length}_" \
                           f"clients={self.federated_config.clients_num}_" \
                           f"alpha={self.federated_config.alpha}"

        cached_file = os.path.join(
            self.data_config.cache_dir, cached_file_name
        )
        return cached_file

    @property
    def prompt_text(self):
        raise NotImplementedError
