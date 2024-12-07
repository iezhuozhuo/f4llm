import os
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from utils.register import registry
from utils.general import run_process
from evals.BaseEvaluator import BaseEvaluator


class DirectoryHandler(FileSystemEventHandler):
    def __init__(self, observer, eval_count, base_opts):
        self.observer = observer
        self.eval_count = eval_count
        self.evaluated_models = 0
        self.logger = registry.get("logger")
        self.base_opts = base_opts

    def on_created(self, event):
        if event.is_directory:
            self.logger.info(f"New checkpoint model: {event.src_path}")
            self.evaluate_directory(event.src_path)
            self.evaluated_models += 1
            if self.evaluated_models >= self.eval_count:
                self.logger.debug("Stop monitoring")
                self.observer.stop()

    def evaluate_directory(self, train_ckp_dir):
        self.logger.info(f"Eval {train_ckp_dir} Start")
        # run pipeline
        eval_cmd = " ".join(self.base_opts)
        eval_cmd += f" --checkpoint_file {train_ckp_dir}"
        # self.logger.debug(eval_cmd)
        run_process(eval_cmd)
        self.logger.info(f"Eval {train_ckp_dir} End")


@registry.register_eval("ayn_local_eval")
class AynLocalBaseEvaluator(BaseEvaluator):
    def __init__(self, *args):
        super().__init__(*args)

        self.base_opts = self.build_eval_cmd()

    def run(self):
        observer = Observer()
        ckp_num = int(self.F.rounds // self.F.save_valid_len)
        eval_path = f"{self.T.save_dir}/{self.T.times}_{self.M.model_type}_lora/"
        run_opts = ["deepspeed", "--include", f"localhost:{self.T.eval_device}",
                    "--master_port", f"{self.T.eval_port}"]
        run_opts += self.base_opts

        event_handler = DirectoryHandler(
            observer=observer,
            eval_count=ckp_num,
            base_opts=run_opts
        )
        observer.schedule(event_handler, eval_path, recursive=False)
        observer.start()

        try:
            while observer.is_alive():
                observer.join(1)
        except KeyboardInterrupt:
            observer.stop()
        observer.join()
