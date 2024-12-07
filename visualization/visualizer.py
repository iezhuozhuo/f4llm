from transformers import TrainerCallback

from utils.register import registry
from visualization.backends import WandbVisBackend, VisBackend

BACKEND_CLS = {
    'wandb': WandbVisBackend
}


class Visualizer(object):
    def __init__(
            self,
            project: str,
            trial_name: str,
            phase: str = 'train',
            config: dict = None,
            key: str = None,
            backend: str = "wandb",
            role: str = 'server',
            client_name=0,
    ):
        self.role = role
        self.phase = phase
        self.trial_name = trial_name
        self.client_name = client_name
        if backend in BACKEND_CLS:
            self.visualizer: VisBackend = BACKEND_CLS[backend](
                key=key, project=project,
                group=trial_name + f'_{phase}',
                run_name=self.role + ('' if self.is_server(self.role) else str(client_name)),
                config=config)
        else:
            raise ValueError(f"backend {backend} not supported!")

    @staticmethod
    def is_server(role_name: str):
        return 'server' in role_name.lower()

    def log(self, global_round: int = -1,
            curr_step: int = -1,
            curr_epoch: float = -1,
            total_step: int = -1,
            data: dict = None):
        id = 'GLOBAL/' if self.is_server(self.role) else f'CLIENT{self.client_name}/'
        self.visualizer.log(id=id,
                            global_round=global_round,
                            curr_step=curr_step,
                            curr_epoch=curr_epoch,
                            total_step=total_step,
                            data=data)


# def rewrite_logs(d):
#     new_d = {}
#     eval_prefix = "eval_"
#     eval_prefix_len = len(eval_prefix)
#     test_prefix = "test_"
#     test_prefix_len = len(test_prefix)
#     for k, v in d.items():
#         if k.startswith(eval_prefix):
#             new_d["eval/" + k[eval_prefix_len:]] = v
#         elif k.startswith(test_prefix):
#             new_d["test/" + k[test_prefix_len:]] = v
#         else:
#             new_d["train/" + k] = v
#     return new_d


class VisCallback(TrainerCallback):

    def __init__(self, vis: Visualizer):
        self.vis = vis

    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        if state.is_world_process_zero and self.vis is not None:
            # logs = rewrite_logs(logs)
            self.vis.log(global_round=registry.get('round'),
                         curr_step=state.global_step,
                         curr_epoch=state.epoch,
                         total_step=state.max_steps,
                         data={**logs, "train/global_step": state.global_step})

    # def on_log(self, args, state, control, model=None, logs=None, **kwargs):
    #     single_value_scalars = [
    #         "train_runtime",
    #         "train_samples_per_second",
    #         "train_steps_per_second",
    #         "train_loss",
    #         "total_flos",
    #     ]
    #     if state.is_world_process_zero:
    #     # for k, v in logs.items():
    #     #     if k in single_value_scalars:
    #     #         self._wandb.run.summary[k] = v
    #         non_scalar_logs = {k: v for k, v in logs.items() if k not in single_value_scalars}
    #         non_scalar_logs = rewrite_logs(non_scalar_logs)
