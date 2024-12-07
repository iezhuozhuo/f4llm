import torch
import transformers
from datasets import Dataset
from dataclasses import dataclass

from utils.register import registry
from utils.general import custom_pad_sequence
from datas.base_data import FedBaseDataManger, FedBaseDataset
from tools.prompts import all_prompts

IGNORE_INDEX = -100


@registry.register_data("rwben")
class RewardBenDataManger(FedBaseDataManger):
    """
    Data manager for Direct Preference Optimization (DPO) datasets
    """

    def __init__(self):
        super().__init__()

        self.phase = registry.get('phase')
        self._set_map()

    def build_inputs(self, prompt_text, text):
        """
        Build inputs for the model

        Args:
            prompt_text: str, the prompt text
            text: str, the text to be processed

        Returns:
            str, the inputs text
        """
        inputs_text = prompt_text.format(text)
        return inputs_text

    def process_examples(self, examples, mode="train", verbose=True):
        """
        Process the examples for DPO, including the prompt, chosen response, and rejected response The inputs are
        formed by concatenating the prompt and the chosen & rejected responses, the labels are also formed in this
        way but with the prompt masked out (with IGNORE_INDEX)

        Args:
            examples: list, the list of examples
            mode: str, the mode of the examples
            verbose: bool, whether to print the verbose

        """
        instances = []
        columns = list(examples[0].keys())
        template = all_prompts[self.data_config.template_name]

        max_model_length = self.data_config.model_max_length
        max_prompt_length = self.data_config.max_prompt_length
        max_response_length = max_model_length - max_prompt_length

        for idx, example in enumerate(examples):
            if 'chosen' not in columns or 'rejected' not in columns:
                assert 'instruction' in columns and 'input' in columns and 'output' in columns
                instruction, input, output = example['instruction'], example['input'], example['output']
                if input is not None and input != "":
                    instruction = instruction + '\n' + input
                assert len(output) > 1
                prompt, chosen, rejected = instruction, output[0], output[1]
            else:
                assert 'prompt' in columns and 'rejected' in columns and 'chosen' in columns
                prompt, chosen, rejected = example['prompt'], example['chosen'], example['rejected']

            source = template.format_map({'Instruction': prompt})
            source_ids = self.tokenizer.encode(text=source, add_special_tokens=False)
            chosen_ids = self.tokenizer.encode(text=chosen, add_special_tokens=False)
            rejected_ids = self.tokenizer.encode(text=rejected, add_special_tokens=False)

            if len(source_ids) > max_prompt_length - 1:
                source_ids = source_ids[:max_prompt_length - 1]
            if len(chosen_ids) > max_response_length - 1:
                chosen_ids = chosen_ids[:max_response_length - 1]
            if len(rejected_ids) > max_response_length - 1:
                rejected_ids = rejected_ids[:max_response_length - 1]

            source_chosen_ids = source_ids + [self.tokenizer.bos_token_id] + chosen_ids + [
                self.tokenizer.eos_token_id]
            source_chosen_labels = [IGNORE_INDEX] * len(source_ids) + [self.tokenizer.bos_token_id] + chosen_ids + [
                self.tokenizer.eos_token_id]
            source_rejected_ids = source_ids + [self.tokenizer.bos_token_id] + rejected_ids + [
                self.tokenizer.eos_token_id]
            source_rejected_labels = [IGNORE_INDEX] * len(source_ids) + [self.tokenizer.bos_token_id] + rejected_ids + [
                self.tokenizer.eos_token_id]

            source_chosen_length, source_rejected_length = len(source_chosen_ids), len(source_rejected_ids)
            max_length = max(source_chosen_length, source_rejected_length)

            source_chosen_ids = source_chosen_ids + [self.tokenizer.pad_token_id] * (
                    max_length - source_chosen_length)
            source_chosen_labels = source_chosen_labels + [IGNORE_INDEX] * (max_length - source_chosen_length)
            source_rejected_ids = source_rejected_ids + [self.tokenizer.pad_token_id] * (
                    max_length - source_rejected_length)
            source_rejected_labels = source_rejected_labels + [IGNORE_INDEX] * (max_length - source_rejected_length)

            inputs_ids = source_chosen_ids + source_rejected_ids
            labels = source_chosen_labels + source_rejected_labels

            instance = {
                "idx": f"{mode}-{idx}",
                "input_ids": inputs_ids, "labels": labels,
            }

            if self.phase != "train":
                if "subset" not in example:
                    raise ValueError
                set_labels = self.get_set_idx(example)
                instance["label_ids"] = set_labels

            instances.append(instance)

        return instances

    def _set_map(self):
        self.set2idx = {set_name: i for i, set_name in enumerate(["Chat", "Chat Hard", "Safety", "Reasoning"])}
        self.idx2set = {i: set_name for set_name, i in self.set2idx.items()}
        registry.register("set2idx", self.set2idx)
        registry.register("idx2set", self.idx2set)

    def get_set_idx(self, dp):
        if dp["subset"] in "alpacaeval-easy, alpacaeval-length, alpacaeval-hard, mt-bench-easy, mt-bench-medium":
            set_num = "Chat"
        elif dp["subset"] in "mt-bench-hard, llmbar-natural, llmbar-adver-neighbor, llmbar-adver-GPTInst, " \
                             "llmbar-adver-GPTOut, llmbar-adver-manual":
            set_num = "Chat Hard"
        elif dp["subset"] in "refusals-dangerous, refusals-offensive, xstest-should-refuse, xstest-should-respond, " \
                             "donotanswer":
            set_num = "Safety"
        elif dp["subset"] in "math-prm, hep-cpp, hep-go, hep-java, hep-js, hep-python, hep-rust":
            set_num = "Reasoning"
        else:
            raise ValueError
        return self.set2idx[set_num]

    def coll_fn(self, model):
        """
        Create pairwise data collator for DPO tasks
        The collator will separate the chosen and rejected responses from the inputs and labels for DPO

        Args:
            model: the model to be used

        Returns:
            Callable, the collator function

        Examples:
            >>> data_collator = data_manager.coll_fn(model)
            >>> dataloader = DataLoader(dataset, collate_fn=data_collator)
            >>> for batch in dataloader:
            >>>   print(batch) # {'chosen_input_ids': ..., 'chosen_labels': ..., 'chosen_attention_mask': ...}
            >>>   break

        """

        @dataclass
        class DataCollatorForPairwiseDataset(object):
            """Collate examples for pairwise dataset."""

            tokenizer: transformers.PreTrainedTokenizer
            phase = registry.get('phase')

            def __call__(self, instances):
                input_ids, chosen_ids, chosen_labels, rejected_ids, rejected_labels = [], [], [], [], []
                set_ids = []
                for instance in instances:
                    length = len(instance["input_ids"]) // 2
                    chosen_id = instance["input_ids"][:length]
                    rejected_id = instance["input_ids"][length:]
                    chosen_label = instance["labels"][:length]
                    rejected_label = instance["labels"][length:]
                    input_ids.append(torch.LongTensor(instance["input_ids"]))
                    chosen_ids.append(torch.LongTensor(chosen_id))
                    chosen_labels.append(torch.LongTensor(chosen_label))
                    rejected_ids.append(torch.LongTensor(rejected_id))
                    rejected_labels.append(torch.LongTensor(rejected_label))
                    if self.phase != "train":
                        set_ids.append(instance["label_ids"])

                input_ids = custom_pad_sequence(input_ids, padding_value=self.tokenizer.pad_token_id,
                                                left_padding=True)
                chosen_input_ids = custom_pad_sequence(chosen_ids, padding_value=self.tokenizer.pad_token_id,
                                                       left_padding=True)
                rejected_input_ids = custom_pad_sequence(rejected_ids, padding_value=self.tokenizer.pad_token_id,
                                                         left_padding=True)
                chosen_labels = custom_pad_sequence(chosen_labels, padding_value=self.tokenizer.pad_token_id,
                                                    left_padding=True)
                rejected_labels = custom_pad_sequence(rejected_labels, padding_value=self.tokenizer.pad_token_id,
                                                      left_padding=True)
                input_data = dict(
                    chosen_input_ids=chosen_input_ids,
                    chosen_labels=chosen_labels,
                    chosen_attention_mask=chosen_input_ids.ne(self.tokenizer.pad_token_id),
                    rejected_input_ids=rejected_input_ids,
                    rejected_labels=rejected_labels,
                    rejected_attention_mask=rejected_input_ids.ne(self.tokenizer.pad_token_id),
                    input_ids=input_ids,
                )
                if self.phase != "train":
                    set_labels = torch.LongTensor(set_ids)
                    input_data["labels"] = set_labels

                return input_data

        data_collator = DataCollatorForPairwiseDataset(tokenizer=self.tokenizer)

        return data_collator
