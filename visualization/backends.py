import logging
from abc import ABC, abstractmethod


def _set_if_gtz(dict, key, value):
    if value is not None and value > 0:
        dict[key] = value


class VisBackend(ABC):

    def __init__(self, key: str, project: str, group: str, run_name: str, config: dict):
        self.key = key
        self.project = project
        self.group = group
        self.run_name = run_name
        self.config = config

    @abstractmethod
    def log(self, id: str, global_round: int, curr_step: int = -1, curr_epoch: float = -1, total_step: int = -1,
            data: dict = None):
        raise NotImplementedError


class WandbVisBackend(VisBackend):

    def __init__(self, key: str, project: str, group: str, run_name: str, config: dict):
        super().__init__(key, project, group, run_name, config)
        import wandb
        self._wandb = wandb
        self._wandb.init(mode='online', project=project, name=run_name, group=group, config=config)

    def log(self,
            id: str,
            global_round: int, curr_step: int = -1, curr_epoch: float = -1, total_step: int = -1,
            data: dict = None):
        # logging.debug(f'{id} Logging  to wandb: {data}')
        prefix = f'{id}_'
        meta_data = {}
        _set_if_gtz(meta_data, 'curr_step', curr_step)
        _set_if_gtz(meta_data, 'total_step', total_step)
        _set_if_gtz(meta_data, 'curr_epoch', curr_epoch)
        _set_if_gtz(meta_data, 'global_round', global_round)
        data.update(meta_data)
        data = {prefix + key: v for key, v in data.items()}
        data['global_round'] = global_round
        self._wandb.log(data)
