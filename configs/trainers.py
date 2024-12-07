from typing import Optional
from dataclasses import dataclass, field
from transformers import TrainingArguments


@dataclass
class TrainArguments(TrainingArguments):
    local_trainer_name: str = field(
        default="fedavg_sft",
        metadata={"help": "The name of the local trainer"},
    )
    times: int = field(
        default=None, metadata={"help": "the identification of each runs."}
    )
    do_reuse: bool = field(
        default=False, metadata={"help": "whether to load last checkpoint"}
    )
    do_sample: bool = field(
        default=False, metadata={"help": "Used for generator"}
    )
    temperature: float = field(
        default=None, metadata={"help": "Used for generator"}
    )
    top_p: float = field(
        default=None, metadata={"help": "Used for generator"}
    )
    generation_max_length: int = field(
        default=1024, metadata={"help": "The number to generate output."}
    )
    save_outputs: bool = field(
        default=False, metadata={"help": "where to save generated output"}
    )
    eval_name: str = field(
        default=None, metadata={"help": "evaluation function"}
    )
    metric_name: str = field(
        default="acc", metadata={"help": "metirc function"}
    )
    loss_name: str = field(
        default=None, metadata={"help": "{xent: cross_entropy}"}
    )
    config_path: str = field(
        default=None, metadata={"help": "the custom yaml file"}
    )
    checkpoint_file: str = field(
        default=None, metadata={"help": "saving the model parameters"}
    )
    is_decreased_valid_metric: bool = field(
        default=False
    )
    do_zst: bool = field(
        default=False
    )
    patient_times: int = field(
        default=-1,
    )
    do_grid: bool = field(
        default=False, metadata={"help": "whether to do grid search"}
    )
    not_overwrite_args: str = field(
        default=None, metadata={"help": "using for preventing overwriting"}
    )
    load_in_8bit: Optional[bool] = field(
        default=False, metadata={"help": "load the model in 8 bits precision"})
    load_in_4bit: Optional[bool] = field(
        default=False, metadata={"help": "load the model in 4 bits precision"})
    eval_reuse: bool = field(
        default=True, metadata={"help": "reusing eval results."}
    )
    test_best: bool = field(
        default=False, metadata={"help": "use best model for test."}
    )
    test_openai: bool = field(
        default=False, metadata={"help": "metric for openai-test."}
    )
    eval_port: str = field(
        default="10001", metadata={"help": "metric for openai-test."}
    )
    eval_device: str = field(
        default="0,1", metadata={"help": "metric for openai-test."}
    )
    eval_during_train: bool =field(
        default=False, metadata={"help": "metric for openai-test."}
    )
    # Used for DPO
    average_log_prob: bool = field(
        default=False, metadata={"help": "used for dpo."}
    )
    reference_free: bool = field(
        default=False, metadata={"help": "used for dpo."}
    )
    dpo_beta: float = field(
        default=0.1, metadata={"help": "used for dpo."}
    )
    # visualization
    visualization: bool = field(
        default=None, metadata={"help": "used for visualization."}
    )
