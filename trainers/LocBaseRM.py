import os
import inspect
from typing import Optional

import torch
from peft import get_peft_model_state_dict
from transformers import Trainer, PreTrainedModel
from transformers.modeling_utils import unwrap_model

from utils.register import registry

IGNORE_INDEX = -100
WEIGHTS_NAME = "adapter_model.bin"
TRAINING_ARGS_NAME = "training_args.bin"


class LocalRMTrainer(Trainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.args.use_last_reward = getattr(self.args, "use_last_reward", False)
        self.args.clm_loss_weight = getattr(self.args, "clm_loss_weight", 0.2)
        # self.logger = registry.get("logger")

    def get_state_dict(self, model):
        pretrained_model_state_dict = model.pretrained_model.state_dict()
        v_head_state_dict = model.v_head.state_dict()
        for k, v in v_head_state_dict.items():
            pretrained_model_state_dict[f"v_head.{k}"] = v
        return pretrained_model_state_dict

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        # logger.info(f"Saving model checkpoint to {output_dir}")

        if not isinstance(self.model, PreTrainedModel):
            if state_dict is None:
                state_dict = self.get_state_dict(self.model)

            if isinstance(unwrap_model(self.model), PreTrainedModel):
                unwrap_model(self.model).save_pretrained(output_dir, state_dict=state_dict)
            else:
                # logger.info("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
                adapter_state_dict = get_peft_model_state_dict(unwrap_model(self.model).pretrained_model)

                # add v_head (v_head not in modules_to_save)
                v_head_state_dict = self.model.v_head.state_dict()
                for k, v in v_head_state_dict.items():
                    adapter_state_dict[f"v_head.{k}"] = v

                torch.save(adapter_state_dict, os.path.join(output_dir, WEIGHTS_NAME))

            try:
                unwrap_model(self.model).pretrained_model.peft_config.save_pretrained(output_dir)
            except AttributeError:
                unwrap_model(self.model).pretrained_model.peft_config['default'].save_pretrained(output_dir)
        else:
            self.model.save_pretrained(output_dir, state_dict=state_dict)

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

    def _set_signature_columns_if_needed(self):
        # important self.label_names/default_label_names != "labels"
        if self._signature_columns is None:
            # Inspect model forward signature to keep only the arguments it accepts.
            signature = inspect.signature(self.model.forward)
            self._signature_columns = list(signature.parameters.keys())
            # self.logger.info(f"The following columns {self._signature_columns} are accepted.")

            if "input_ids" not in self._signature_columns:
                self._signature_columns += ["input_ids"]

            # Labels may be named label or label_ids, the default data collator handles that.
            self._signature_columns += list(set(["label", "label_ids", "labels"] + self.label_names))
            # self.logger.info(f"The following columns {self._signature_columns} are accepted.")

    def compute_loss(self, model, inputs, return_outputs=False):

        _, chosen_clm_loss, chosen_value = model(
            input_ids=inputs["chosen_input_ids"], attention_mask=inputs["chosen_attention_mask"],
            labels=inputs["chosen_labels"], return_dict=True)
        _, _, rejected_value = model(
            input_ids=inputs["rejected_input_ids"], attention_mask=inputs["rejected_attention_mask"], return_dict=True)

        chosen_labels, rejected_labels = inputs["chosen_labels"], inputs["rejected_labels"]
        chosen_action_masks = chosen_labels.ne(IGNORE_INDEX).long()
        rejected_action_masks = rejected_labels.ne(IGNORE_INDEX).long()

        chosen_value = chosen_value * chosen_action_masks
        rejected_value = rejected_value * rejected_action_masks

        batch_size = chosen_value.shape[0]
        chosen_seq_lengths = (torch.ne(inputs["chosen_input_ids"], self.tokenizer.pad_token_id).sum(-1) - 1).to \
            (chosen_value.device)
        rejected_seq_lengths = (torch.ne(inputs["rejected_input_ids"], self.tokenizer.pad_token_id).sum(-1) - 1).to \
            (rejected_value.device)

        chosen_end_token_value = chosen_value[
            torch.arange(batch_size, device=chosen_value.device), chosen_seq_lengths]
        rejected_end_token_value = rejected_value[
            torch.arange(batch_size, device=rejected_value.device), rejected_seq_lengths]

        if self.args.use_last_reward:
            loss1 = -torch.nn.functional.logsigmoid(chosen_end_token_value - rejected_end_token_value).mean()
        else:

            loss1 = -torch.nn.functional.logsigmoid(chosen_value - rejected_value).mean()

        loss2 = self.args.clm_loss_weight * chosen_clm_loss
        loss = loss1 + loss2

        outputs = dict(
            chosen_end_token_value=chosen_end_token_value,  # shape: (batch_size,)
            rejected_end_token_value=rejected_end_token_value,  # shape: (batch_size,)
        )

        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys):
        inputs = self._prepare_inputs(inputs)

        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        with torch.no_grad():
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
        loss = loss.detach()

        logits = tuple(v for k, v in outputs.items() if k in ["chosen_end_token_value", "rejected_end_token_value"])
        if prediction_loss_only:
            return (loss, None, None)

        logits = torch.stack(logits, dim=1)
        labels = torch.zeros(logits.shape[0]).to(logits.device)

        return loss, logits, labels
