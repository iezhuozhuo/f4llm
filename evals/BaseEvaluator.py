import os
import pandas as pd
from glob import glob
from tabulate import tabulate

from peft import load_peft_weights
from trainers.LocBaseSFT import LocalSFTTrainer

from visualization.visualizer import Visualizer
from utils.register import registry
from utils.general import is_best, pickle_read, run_process
from utils.general import setup_seed, read_json
from trainers.BaseEngine import BaseEngine


class BaseEvaluator(BaseEngine):
    def __init__(self, *args):
        super().__init__(*args)

    def on_eval_before(self):
        # set seed
        setup_seed(self.T.seed)

        # build dataset and dataloader
        self._build_data()

        # build federated model
        self._build_model()

        # build metric
        self._build_metric()  # return computer metric

        # global model
        self.best_glo_params = self.serialize_model_parameters()

        # build visualizer
        self._build_visualizer()

    def _load_examples(self):
        if 'json' in self.D.eval_data_path:
            examples = read_json(self.D.eval_data_path)
        elif 'pkl' in self.D.eval_data_path:
            examples = pickle_read(self.D.eval_data_path)
        else:
            raise TypeError
        return examples

    def _build_data(self):
        self.logger.info(f"{self.role} building dataset ...")
        self.data = registry.get_data_class(self.D.data_name)()
        if self.D.eval_data_path is not None:
            # load examples used for open-ai test
            examples = self._load_examples()
            self.logger.debug(f"Custom eval path [{self.D.eval_data_path}] with {len(examples)} examples")
            self.data.train_dataset_dict[-1] = None
            self.data.test_dataset_dict[-1] = self.data.process_examples(examples, self.phase)
            self.data.valid_dataset_dict[-1] = self.data.process_examples(examples, self.phase)
        else:
            self.data.load_data()

    def run(self):
        self.on_eval_before()
        self.on_eval()
        self.on_eval_end()

    def on_eval(self):
        if self.phase == "zst":
            self.zero_test()
            return

        # default local test
        best_file = None
        single_file = False

        pattern_name = "round*" if self.is_fl else "steps*"
        pattern = os.path.join(self.T.checkpoint_file, pattern_name)
        checkpoint_files = sorted(glob(pattern, recursive=True),
                                  key=lambda x: os.path.getctime(x), reverse=False)
        if len(checkpoint_files) == 0:
            # single checkpoint test
            self.logger.debug("Eval Single Checkpoint")
            checkpoint_files = [self.T.checkpoint_file]
            single_file = True

        ckpt_metric = {}
        for checkpoint_file in checkpoint_files:
            file = checkpoint_file.split("/")[-1]
            self.logger.info(f"Eval {file} Start")

            eval_key = registry.get("eval_key", "train")
            _, valid_dataset = self.get_dataset(-1, eval_key)

            checkpoint_opt_file = os.path.join(checkpoint_file, f"{file}.result.pkl")
            registry.register('checkpoint_opt_file', checkpoint_opt_file)

            if self.T.eval_reuse and os.path.exists(checkpoint_opt_file):
                eval_preds = pickle_read(checkpoint_opt_file)
                valid_metric = self.metric.calculate_metric(eval_preds)["result"]
            else:
                valid_metric = self.eval_fun(
                    eval_dataset=valid_dataset, checkpoint_file=checkpoint_file)["eval_result"]

            metric_value = valid_metric[self.metric_name]
            if is_best(self.global_valid_best_metric, metric_value, self.metric.is_decreased_valid_metric):
                self.global_valid_best_metric = metric_value
                self.best_glo_params = self.serialize_model_parameters()
                best_file = file

            ckpt_metric[file] = valid_metric
            self.logger.info(f"Model: {file}, Metric: {metric_value:.3f}, "
                             f"Best Model: {best_file}, "
                             f"Best: {self.global_valid_best_metric:.3f}")
            self.logger.info(f"Eval Results save in {checkpoint_opt_file}")

            # self._build_model()

        # Not that I set single_file to False to test wandb
        # single_file = False
        if not single_file and not self.T.test_openai:
            metric_path = os.path.join(self.T.checkpoint_file, "metric.csv")
            metrics_df = pd.DataFrame.from_dict(ckpt_metric, orient='index')
            metrics_df["mean"] = metrics_df.mean(axis=1).round(3)
            sorted_metrics_df = metrics_df.sort_values(by='mean', ascending=False)
            sorted_metrics_df.reset_index(inplace=True)
            sorted_metrics_df.columns = ["round"] + list(sorted_metrics_df.columns[1:])

            sorted_metrics_df.to_csv(metric_path, sep="\t", index=False)
            self.logger.info(f"Metric outputs saved to {metric_path}")

            self.logger.info(f"\n{tabulate(sorted_metrics_df, headers='keys', tablefmt='pretty', showindex=False)}")
            self.logger.info(f"Copy Results")
            with self.T.main_process_first():
                run_process(f"cat {metric_path}")

            if self.visualizer is not None:
                self.visualizer.log(
                    # global_round=registry.get('round'),
                    # global_round=self.round,
                    data={
                        "test_loss": ckpt_metric[self.T.checkpoint_file.split("/")[-1]][self.metric_name]
                    },
                    global_round=int(self.T.checkpoint_file.split("/")[-1].split("-")[-1])
                )

        if self.T.test_best and not self.T.test_openai:
            self.logger.critical("Test Start")
            self.deserialize_model(self.best_glo_params)

            _, test_dataset = self.get_dataset(-1, "test")
            test_result = self.eval_fun(test_dataset)

            global_test_best_metric = test_result["eval_result"]
            self.global_test_best_metric = ""
            for metric_name, metric in global_test_best_metric.items():
                self.global_test_best_metric += f"{metric_name}={metric:.3f}_"
            self.global_test_best_metric = self.global_test_best_metric[0:-1]

            self.logger.critical(f"Test Done, "
                                 f"Checkpoint Metric: {self.global_test_best_metric}, "
                                 f"Model Path: {self.T.checkpoint_file}")

            # if os.path.isdir(self.T.checkpoint_opt_file):
            #     self.metric_save()

    def build_eval_op(self, model=None):
        # Initialize Eval Trainer
        raise NotImplementedError

    def eval_fun(self, eval_dataset, checkpoint_file=None):
        if checkpoint_file is not None:
            ckt_param = load_peft_weights(checkpoint_file)
            self.deserialize_model(ckt_param)
        # merge_model = self.model.merge_and_unload()
        # eval_op = self.build_eval_op(merge_model)
        eval_op = self.build_eval_op()
        eval_result = eval_op.evaluate(
            eval_dataset,
        )
        del eval_op
        return eval_result

    def zero_test(self):

        self.deserialize_model(self.best_glo_params)
        _, test_dataset = self.get_dataset(-1, "test")
        test_metric = self.eval_fun(
            eval_dataset=test_dataset)["eval_result"]

        global_test_best_metric = ""
        for metric_name, metric in test_metric.items():
            global_test_best_metric += f"{metric_name}={metric:.3f}_"
        self.global_test_best_metric = global_test_best_metric[0:-1]

        self.logger.debug(f"zst metric: {self.global_test_best_metric}")
        self.logger.debug(f"zst output file: {self.T.checkpoint_opt_file}")

    def on_eval_end(self):
        # local test: save metric and results
        # self.metric_save()
        # openai test:
        # build openai test
        ...
