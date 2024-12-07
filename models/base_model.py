from abc import ABC

import torch
from peft import (TaskType, get_peft_model, prepare_model_for_kbit_training,
                  LoraConfig, PrefixTuningConfig)
from transformers import AutoConfig, BitsAndBytesConfig, AutoModelForCausalLM, AutoModel

from utils.general import get_parameter_number
from utils.general import is_petuning
from utils.register import registry

"""
This module contains the base class 'BaseModels' for all pre-trained models in the project, it provides the basic 
structure and methods for building and configuring language models. 

Example Usage:
    from models.base_model import BaseModels
    from utils.register import registry
    @registry.register_model("llama2-chat")
    class LlaMa2Model(BaseModels):
        ...
"""


class BaseModels(ABC):
    """
    Base class for all pre-trained language models

    Attributes:
        task_name (str): Name of the task
        model_config (configs.ModelArguments): Model configuration
        train_config (configs.TrainingArguments): Training configuration
        logger (Logger): Logger object
        auto_config (AutoConfig): AutoConfig object for the model
    """

    def __init__(self, task_name):
        super().__init__()

        config = registry.get("config")
        self.model_config = config.model_config
        self.train_config = config.training_config

        self.role = config.F.role
        self.task_name = task_name
        self.phase = registry.get('phase')
        self.logger = registry.get("logger")

        self._build_config()

    def _build_config(self):
        """
        Build the AutoConfig object for the model, quantize if needed
        Returns:
            AutoConfig: AutoConfig object for the model
        """
        self.auto_config = AutoConfig.from_pretrained(
            self.model_config.model_name_or_path,
            trust_remote_code=True,
        )

        if self.phase == 'train' and (self.train_config.load_in_8bit or self.train_config.load_in_4bit):
        # if (self.train_config.load_in_8bit or self.train_config.load_in_4bit):
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=self.train_config.load_in_8bit, load_in_4bit=self.train_config.load_in_4bit
            )
        else:
            quantization_config = None

        torch_dtype = torch.bfloat16  # important

        if self.role == "server":
            device_map = "cpu"
        else:
            device_map = None

        self.extra_config = {
            "device_map": device_map,
            "torch_dtype": torch_dtype,
            "quantization_config": quantization_config
        }

    def build_model(self):
        backbone = self._add_base_model()

        backbone = self._add_quantize_model(backbone)

        if is_petuning(self.model_config.tuning_type):
            backbone = self._add_delta_model(backbone)

        return backbone

    def _add_base_model(self):
        backbone = AutoModelForCausalLM.from_pretrained(
            self.model_config.model_name_or_path,
            config=self.auto_config,
            trust_remote_code=True,
            **self.extra_config
        )
        return backbone

    def _add_delta_model(self, backbone):
        """
        Convert the backbone model to Delta model (PETuning)

        Args:
            backbone (PreTrainedModel): backbone language model

        Returns:
            PreTrainedModel: Pre-trained model backbone with Delta configured
        """
        if is_petuning(self.model_config.tuning_type):
            if "lora" in self.model_config.tuning_type:
                target_modules = getattr(self.model_config, "target_modules", self.target_modules)
                peft_config = LoraConfig(task_type=self.task_type,
                                         r=self.model_config.lora_rank,
                                         lora_alpha=self.model_config.lora_alpha,
                                         lora_dropout=self.model_config.lora_dropout,
                                         target_modules=target_modules)

            elif "prefix" in self.model_config.tuning_type:
                peft_config = PrefixTuningConfig(task_type=self.task_type,
                                                 num_virtual_tokens=self.model_config.num_virtual_tokens)
            else:
                raise NotImplementedError(f"NotImplemented tuning_type: {self.model_config.tuning_type}")
            backbone = get_peft_model(backbone, peft_config)
        else:
            raise NotImplementedError(f"NotImplemented tuning_type: {self.model_config.tuning_type}")

        self.logger.debug(f"Delta Model: {self.model_config.tuning_type}, "
                          f"Parameters: {get_parameter_number(backbone)} M")

        return backbone

    def _add_quantize_model(self, backbone):
        """
        Quantize the backbone model

        Args:
            backbone (PreTrainedModel): backbone language model
        Returns:
            PreTrainedModel: Pre-trained model backbone with quantization configured
        """
        if self.phase == 'train' and (self.train_config.load_in_8bit or self.train_config.load_in_4bit):
        # if (self.train_config.load_in_8bit or self.train_config.load_in_4bit):
            self.logger.info(f"Quantized to 8bit")
            backbone = prepare_model_for_kbit_training(
                backbone, use_gradient_checkpointing=self.train_config.gradient_checkpointing
            )

        return backbone

    @property
    def task_type(self):
        return TaskType.CAUSAL_LM

    @property
    def target_modules(self):
        """
        Target modules for PETuning

        Returns:
            List[str]: List of target modules' names
        """
        return None
