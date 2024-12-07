import copy
from dataclasses import dataclass

import transformers

from datas.base_data import FedBaseDataManger
from tools.prompts import all_prompts
from utils.general import custom_pad_sequence
from utils.register import registry

"""
This module contains DataManagers for Supervised Fine-tuning (SFT) tasks. Supervised Fine-tuning (SFT) is a task 
where the model is fine-tuned on a dataset with supervision signals, take QA as an example, the model is trained to 
generate the answer given the question. For mainstream auto-regressive generative language models (e.g., GPT2, Llama, 
etc.), the input is concatenated with the supervision signal, and the model is trained to generate the target 
sequence.
 
"""
IGNORE_INDEX = -100


def _tokenize_fn(strings, tokenizer, mode='train'):
    """
    Tokenize the strings for Auto-regressive Supervised Fine-tuning (SFT) tasks.
    Args:
        strings (List[str]): List of strings to tokenize.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer object.
        mode (str): Task mode. Default is 'train'. Train mode will truncate the input to the model's max length.

    Returns:
        dict: Dictionary containing the tokenized input_ids, labels, input_ids_lens, and labels_lens.

    Examples:
        >>> _tokenize_fn(["Hello, world!", "How are you?"], tokenizer)
        {'input_ids': [[101, 7592, 1010, 2088, 999, 102], [101, 2129, 2024, 2017, 1029, 102]],
         'labels': [[-100, 7592, 1010, 2088, 999, 102], [-100, 2129, 2024, 2017, 1029, 102]],
         'input_ids_lens': [6, 6], 'labels_lens':
    """
    truncation = True if mode == "train" else False
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,  # usable when mode == train
            truncation=truncation,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
        sources,
        targets,
        tokenizer,
        mode="train"
):
    """
    Preprocess the data by tokenizing.

    Args:
        sources (List[str]): List of source strings.
        targets (List[str]): List of target strings (supervision signal).
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer object.
        mode (str): Task mode. Default is 'train', which will concatenate the source and target strings as training data
            and mask out the source part in the labels.
    Returns:
        dict: Dictionary containing the tokenized input_ids and labels.

    Examples:
        >>> preprocess(["Hello, world!", "How are you?"], ["I am fine.", "Thank you."], tokenizer, mode="train")
        {'input_ids': [[101, 7592, 1010, 2088, 999, 102, 1045, 2572, 2986, 1012, 102], [101, 2129, 2024, 2017, 1029, 102, 4067, 2017, 1012, 102]],
            'labels': [[-100, 7592, 1010, 2088, 999, 102, 1045, 2572, 2986, 1012, 102], [-100, 2129, 2024, 2017, 1029, 102, 4067, 2017, 1012,
            102]]}
        >>> preprocess(["Hello, world!", "How are you?"], ["I am fine.", "Thank you."], tokenizer, mode="test")
        {'input_ids': [[101, 7592, 1010, 2088, 999, 102, 1045, 2572, 2986, 1012, 102], [101, 2129, 2024, 2017, 1029, 102, 4067, 2017, 1012, 102]],
            'labels': [[101, 1045, 2572, 2986, 1012, 102], [101, 4067, 2017, 1012, 102]]}

    """
    if mode == "train":
        examples = [s + t for s, t in zip(sources, targets)]
        examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
        input_ids = examples_tokenized["input_ids"]
        labels = copy.deepcopy(input_ids)
        for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
            label[:source_len] = IGNORE_INDEX
    else:
        sources_tokenized = _tokenize_fn(sources, tokenizer)
        targets_tokenized = _tokenize_fn(targets, tokenizer, mode)  # fix truncated label
        input_ids = sources_tokenized["input_ids"]
        labels = targets_tokenized["input_ids"]
    return dict(input_ids=input_ids, labels=labels)


@registry.register_data("llama_sft")
class LlaMaGenDataManger(FedBaseDataManger):
    """
    Data manager for supervised fine-tuning tasks on Llama (Decoder-only, auto-regressive).
    """

    def __init__(self):
        super().__init__()

    def build_inputs(self, prompt_text, text):
        """
        Build inputs for the model.
        Args:
            prompt_text (str): Prompt text.
            text (str): Raw text to be processed.
        Returns:
            str: Inputs text with prompt embedded.
        """
        inputs_text = prompt_text.format(text)
        return inputs_text

    def process_examples(self, examples, mode="train", verbose=True):
        """
        Process examples for supervised fine-tuning tasks.

        Args:
            examples (List[dict]): List of examples.
            mode (str): Task mode. Default is 'train'.
            verbose (bool): Whether to print the first example for debugging.

        Returns:
            List[dict]: List of processed examples.

        """
        instances = []
        PROMPT_DICT = all_prompts[self.data_config.template_name]
        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]

        sources = [
            prompt_input.format_map(example) if example.get("input",
                                                            "<noinput>") != "<noinput>" else prompt_no_input.format_map(
                example)
            for example in examples
        ]

        if mode == "train":
            target_key = 'response'
        # elif self.data_config.task_name in ["medsi"]:
        #     target_key = 'gpt-4-answer'
        elif self.data_config.task_name in ["superni", "medi", "alpt", "tde_alpt"]:
            target_key = 'response'
        elif self.data_config.task_name in ['safe_rlhf']:
            target_key = 'chosen'
        else:
            target_key = 'output'

        targets = [f"{example[target_key]}{self.tokenizer.eos_token}" for example in examples]

        # if registry.get("round", 0) == 0:
        if mode == "train" and verbose:
            self.logger.info("=" * 40)
            self.logger.info(f"{mode} 0: {sources[0]} {targets[0]}")
            self.logger.info("=" * 40)

        data_dict = preprocess(sources, targets, self.tokenizer, mode)
        for idx, input_ids in enumerate(data_dict["input_ids"]):
            instances.append(
                {"input_ids": input_ids, "labels": data_dict["labels"][idx],
                 "idx": f"{mode}-{idx}", "example": examples[idx]}
            )
        return instances

    def coll_fn(self, model):
        """
        Build data collator for supervised fine-tuning tasks.
        Args:
            model (transformers.PreTrainedModel): Model object.
        Returns:
            Callable: Data collator function.
        """

        @dataclass
        class DataCollatorForSupervisedDataset(object):
            """Collate examples for supervised fine-tuning."""

            tokenizer: transformers.PreTrainedTokenizer

            def __call__(self, instances):
                left_padding = False if registry.get("phase") == "train" else True

                input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
                input_ids = custom_pad_sequence(input_ids, padding_value=self.tokenizer.pad_token_id,
                                                left_padding=left_padding)
                labels = custom_pad_sequence(labels, padding_value=self.tokenizer.pad_token_id,
                                             left_padding=left_padding)

                return dict(
                    input_ids=input_ids,
                    labels=labels,
                    attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
                )

        data_collator = DataCollatorForSupervisedDataset(tokenizer=self.tokenizer)

        return data_collator
