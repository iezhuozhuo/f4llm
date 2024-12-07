import torch
import inspect
from torch import nn
from peft import PeftModel
import torch.nn.functional as F
from transformers import Trainer
from typing import Any, Dict, List, Optional, Tuple, Union

IGNORE_INDEX = -100


class LocalDPOTrainer(Trainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_log_probs(self, logits, labels):
        log_probs = F.log_softmax(logits, dim=-1)
        log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1))
        return log_probs_labels.squeeze(-1)

    def masked_mean(self, data, mask, dim=None, eps=1e-8):
        data = data * mask
        if dim is not None:
            return data.sum(dim=dim) / (mask.sum(dim=dim) + eps)
        else:
            return data.sum() / (mask.sum() + eps)

    def get_entropy(self, logits, mask):
        probs = torch.nn.functional.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = self.masked_mean(-torch.sum(probs * log_probs, dim=-1), mask)
        return entropy

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

    def get_model_output(self, model, inputs, is_ref_model=False):

        if is_ref_model:
            if isinstance(model, nn.parallel.DistributedDataParallel):
                with model.module.disable_adapter():
                    chosen_logits = model(input_ids=inputs["chosen_input_ids"],
                                          attention_mask=inputs["chosen_attention_mask"], return_dict=True).logits
                    rejected_logits = model(input_ids=inputs["rejected_input_ids"],
                                            attention_mask=inputs["rejected_attention_mask"], return_dict=True).logits
            elif isinstance(model, PeftModel):
                with model.disable_adapter():
                    chosen_logits = model(input_ids=inputs["chosen_input_ids"],
                                          attention_mask=inputs["chosen_attention_mask"], return_dict=True).logits
                    rejected_logits = model(input_ids=inputs["rejected_input_ids"],
                                            attention_mask=inputs["rejected_attention_mask"], return_dict=True).logits
            else:
                raise AttributeError(
                    f" model object [{model.__class__.__name__}] has no attribute [disable_adapter] "
                )
        else:
            chosen_logits = model(input_ids=inputs["chosen_input_ids"],
                                  attention_mask=inputs["chosen_attention_mask"], return_dict=True).logits
            rejected_logits = model(input_ids=inputs["rejected_input_ids"],
                                    attention_mask=inputs["rejected_attention_mask"], return_dict=True).logits

        chosen_labels, rejected_labels = inputs["chosen_labels"], inputs["rejected_labels"]
        chosen_action_masks = chosen_labels.ne(IGNORE_INDEX).long()
        rejected_action_masks = rejected_labels.ne(IGNORE_INDEX).long()

        chosen_log_probs = self.get_log_probs(chosen_logits[:, :-1, :], inputs["chosen_input_ids"][:, 1:])
        rejected_log_probs = self.get_log_probs(rejected_logits[:, :-1, :], inputs["rejected_input_ids"][:, 1:])

        if self.args.average_log_prob:
            chosen_logps = self.masked_mean(chosen_log_probs, chosen_action_masks[:, 1:], dim=-1)
            rejected_logps = self.masked_mean(rejected_log_probs, rejected_action_masks[:, 1:], dim=-1)
        else:
            chosen_logps = (chosen_log_probs * chosen_action_masks[:, 1:]).sum(dim=-1)
            rejected_logps = (rejected_log_probs * rejected_action_masks[:, 1:]).sum(dim=-1)

        chosen_entropy = self.get_entropy(chosen_logits[:, :-1, :], chosen_action_masks[:, 1:])
        rejected_entropy = self.get_entropy(rejected_logits[:, :-1, :], rejected_action_masks[:, 1:])
        return chosen_entropy, rejected_entropy, chosen_logps, rejected_logps

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):

        chosen_entropy, rejected_entropy, chosen_logps, rejected_logps = self.get_model_output(model, inputs)
        with torch.no_grad():
            ref_chosen_entropy, ref_rejected_entropy, ref_chosen_logps, ref_rejected_logps = self.get_model_output \
                (model, inputs, is_ref_model=True)

        chosen_ratio = self.args.dpo_beta * (chosen_logps - ref_chosen_logps)
        rejected_ratio = self.args.dpo_beta * (rejected_logps - ref_rejected_logps)

        pi_ratio = self.args.dpo_beta * (chosen_logps - rejected_logps)
        ref_ratio = self.args.dpo_beta * (ref_chosen_logps - ref_rejected_logps)

        if self.args.reference_free:
            ref_ratio = 0

        loss = -torch.nn.functional.logsigmoid(pi_ratio - ref_ratio).mean()

        outputs = dict(
            chosen_reward=chosen_ratio.detach(),  # shape: (batch_size,)
            rejected_reward=rejected_ratio.detach(),
            pi_ratio=pi_ratio.detach(),
            ref_ratio=ref_ratio.detach(),
            chosen_entropy=chosen_entropy.detach(),
            rejected_entropy=rejected_entropy.detach(),
            ref_chosen_entropy=ref_chosen_entropy.detach(),
            ref_rejected_entropy=ref_rejected_entropy.detach(),
            chosen_ce_loss=-chosen_logps.detach(),
            rejected_ce_loss=-rejected_logps.detach(),
        )

        return (loss, outputs) if return_outputs else loss

    def prediction_step(self,
                        model: nn.Module,
                        inputs: Dict[str, Union[torch.Tensor, Any]],
                        prediction_loss_only: bool,
                        ignore_keys: Optional[List[str]] = None,
                        ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:

        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []
        with torch.no_grad():
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
        loss = loss.detach()
        logits = tuple(v for k, v in outputs.items() if k in ["chosen_reward", "rejected_reward"])
        if prediction_loss_only:
            return loss, None, None
        logits = torch.stack(logits, dim=1)

        # chosen first, therefore, we can use label_ids as the label for rewardben
        if "labels" in inputs:
            labels = inputs["labels"]  # set_name info
        else:
            labels = torch.zeros(logits.shape[0]).to(logits.device)
        return loss, logits, labels
