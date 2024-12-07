from models.base_model import BaseModels
from utils.register import registry

"""
This module contains the model classes for the basic foundation LLMs in this project. The models are registered in the model registry for easy access and configuration. Each model class inherits from the BaseModels class, which provides common methods and properties for the models.

Example Usage:
    from models.fundations import ChatGLModel
    from utils.register import registry
    @registry.register_model("chatglm")
"""

@registry.register_model("chatglm")
class ChatGLModel(BaseModels):
    def __init__(self, task_name):
        super().__init__(task_name)


@registry.register_model("baichuan")
class BaiChuanModel(BaseModels):
    def __init__(self, task_name):
        super().__init__(task_name)

    @property
    def target_modules(self):
        return ["W_pack"]

@registry.register_model("llama2-chat")
@registry.register_model("tinyllama")
@registry.register_model("llama2-base")
class LlaMa2Model(BaseModels):
    def __init__(self, task_name):
        super().__init__(task_name)


@registry.register_model("qwen")
class QwenModel(BaseModels):
    def __init__(self, task_name):
        super().__init__(task_name)

    @property
    def target_modules(self):
        return ["c_attn", "c_proj", "w1", "w2"]


@registry.register_model("llama2-rm")
class LlaMa2Model(BaseModels):
    def __init__(self, task_name):
        super().__init__(task_name)

    def build_model(self):
        from utils.general import is_petuning
        from trl import AutoModelForCausalLMWithValueHead

        backbone = self._add_base_model()

        backbone = self._add_quantize_model(backbone)

        if is_petuning(self.model_config.tuning_type):
            backbone = self._add_delta_model(backbone)

        backbone = AutoModelForCausalLMWithValueHead.from_pretrained(backbone)

        return backbone
