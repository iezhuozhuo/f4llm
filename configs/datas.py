from dataclasses import dataclass, field
from typing import Optional
import os


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    lang_id: Optional[str] = field(default=None, metadata={"help": "Language id for XLMs."})

    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the task to use (via the datasets library). [seq2seq]"}
    )
    raw_dataset_path: Optional[str] = field(
        default=None, metadata={"help": "The path of the train/valid/test raw data (.pkl)"}
    )
    partition_dataset_path: Optional[str] = field(
        default=None, metadata={"help": "The federated partition path of the raw data (.pkl)"}
    )
    eval_data_path: Optional[str] = field(
        default=None, metadata={"help": "The extra eval data path (.pkl)"}
    )
    llm_eval_name: Optional[str] = field(
        default="alpaca", metadata={"help": "The eval name for llm-eval"}
    )
    data_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    template_name: Optional[str] = field(
        default="llama_alpaca", metadata={"help": "The name of the data template for tuning."}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "The save dir of the tokenized dataset."}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    debug_mode: bool = field(
        default=False, metadata={"help": "whether to use debug mode"}
    )
    model_max_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    max_prompt_length: Optional[int] = field(
        default=128,
        metadata={
            "help": (
                "The maximum total prompt length after tokenization. Sequences longer "
                "than this will be truncated (including instruction)."
            )
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total sequence length for validation target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
                "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
                "during ``evaluate`` and ``predict``."
            )
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to model maximum sentence length. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                "efficient on GPU but very bad for TPU."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    num_beams: Optional[int] = field(
        default=1,
        metadata={
            "help": (
                "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
                "which is used during ``evaluate`` and ``predict``."
            )
        },
    )
    ignore_index: int = field(
        default=-100,
        metadata={
            "help": "using for ignore token index."
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    forced_bos_token: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The token to force as the first generated token after the decoder_start_token_id."
                "Useful for multilingual models like mBART where the first generated token"
                "needs to be the target language token (Usually it is the target language token)"
            )
        },
    )


    def __post_init__(self):

        if self.task_name is None:
            raise ValueError(f"The task_name must be set, but {self.task_name} found")
        else:
            self.task_name = self.task_name.lower()

        if self.raw_dataset_path is None:
            raise ValueError(f"The raw_dataset_path must be set, but {self.raw_dataset_path} found")
        else:
            if not self.raw_dataset_path.endswith(".pkl"):
                self.raw_dataset_path = os.path.join(
                    self.raw_dataset_path, f"{self.task_name}_data.pkl"
                )

        if self.partition_dataset_path is None:
            raise ValueError(f"The raw_dataset_path must be set, but {self.raw_dataset_path} found")
        else:
            if not self.partition_dataset_path.endswith(".pkl"):
                self.partition_dataset_path = os.path.join(
                    self.partition_dataset_path, f"{self.task_name}_partition.pkl"
                )

        if self.data_name is None:
            self.data_name = self.task_name
        else:
            self.data_name = self.data_name.lower()

        if self.val_max_target_length is None:
            self.val_max_target_length = self.model_max_length
