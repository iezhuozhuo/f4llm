import os
from multiprocessing import Pool

import pandas as pd
from glob import glob
from tabulate import tabulate

from utils.register import registry
from evals.BaseEvaluator import BaseEvaluator
from utils.general import setup_seed, read_json, run_process

task_group = {
    "alpaca": "winogrande,ai2_arc,hellaswag,truthfulqa_mc2,mmlu",
    "rewardben": "Chat,Chat Hard,Safety,Reasoning",
}

task_mertic_mapping = {
    "winogrande": "acc,none",
    'truthfulqa_mc2': 'acc,none',
    'mmlu': 'acc,none',
    "hellaswag": "acc_norm,none",
    "ai2_arc": "acc_norm,none",
    # "":
}


def get_task(task_name):
    if task_name in ["alpaca", "alpacare"]:
        tasks = task_group[task_name].split(",")
    else:
        # single task eval
        tasks = task_name.split(",")
    return tasks


def merge_metric(result_files, tasks, output_path, task_name, logger):
    mean_metrics = {}
    best_metrics = {}
    for result_file in result_files:
        checkpoint_name = result_file.split("/")[-2]
        mean_metrics[checkpoint_name] = {}
        # checkpoint_path = "/".join(result_file.split("/")[0:-1])
        # logger.info(f">> Test result_file: {checkpoint_name}")

        metrics = read_json(result_file)["results"]
        for task in tasks:
            metric_name = getattr(task_mertic_mapping, task, "accuracy")
            metric = round(metrics[task][metric_name]*100, 1)
            mean_metrics[checkpoint_name][task] = metric
            if task not in best_metrics or best_metrics[task] < metric:
                best_metrics[task] = metric

    metrics_df = pd.DataFrame.from_dict(mean_metrics, orient='index')
    metrics_df["mean"] = metrics_df.mean(axis=1).round(1)
    sorted_metrics_df = metrics_df.sort_values(by='mean', ascending=False)
    sorted_metrics_df.reset_index(inplace=True)
    sorted_metrics_df.columns = ["round"] + list(sorted_metrics_df.columns[1:])

    all_metric_path = os.path.join(output_path, f"{task_name}_round_metric.csv")
    sorted_metrics_df.to_csv(all_metric_path, sep="\t", index=False)
    logger.info(f">> Metric outputs saved to {all_metric_path}")

    best_metric_path = os.path.join(output_path, f"{task_name}_best_metric.csv")
    best_metric_df = pd.DataFrame.from_dict({"best": best_metrics}, orient='index')
    best_metric_df["mean"] = best_metric_df.mean(axis=1).round(1)
    # best_metric_df.reset_index(inplace=True)
    best_metric_df.to_csv(best_metric_path, sep="\t", index=False)
    logger.info(f">> Best Metric outputs saved to {best_metric_path}")

    logger.info(f'\n{tabulate(sorted_metrics_df, headers="keys", tablefmt="pretty", showindex=False)}')
    logger.info(f'\n{tabulate(best_metric_df, headers="keys", tablefmt="pretty", showindex=False)}')
    logger.info(f">> Copy Results")
    run_process(f"cat {all_metric_path}")
    run_process(f"cat {best_metric_path}")


def single_metric(result_file, tasks, output_path, task_name, logger):
    mean_metrics = {}
    checkpoint_name = result_file.split("/")[-2]
    mean_metrics[checkpoint_name] = {}

    metrics = read_json(result_file)["results"]
    for task in tasks:
        metric_name = task_mertic_mapping[task]
        metric = round(metrics[task][metric_name] * 100, 1)
        mean_metrics[checkpoint_name][task] = metric

    metrics_df = pd.DataFrame.from_dict(mean_metrics, orient='index')
    metrics_df["mean"] = metrics_df.mean(axis=1).round(1)
    sorted_metrics_df = metrics_df.sort_values(by='mean', ascending=False)
    sorted_metrics_df.reset_index(inplace=True)
    sorted_metrics_df.columns = ["round"] + list(sorted_metrics_df.columns[1:])

    metric_path = os.path.join(output_path, f"{task_name}_metric.csv")
    sorted_metrics_df.to_csv(metric_path, sep="\t", index=False)
    logger.info(f">> Metric outputs saved to {metric_path}")

    logger.info(f'\n{tabulate(sorted_metrics_df, headers="keys", tablefmt="pretty", showindex=False)}')
    logger.info(f">> Copy Results")
    run_process(f"cat {metric_path}")


@registry.register_eval("llm-eval")
class BenEvaluator(BaseEvaluator):
    def __init__(self, *args):
        super().__init__(*args)

    def on_eval_before(self):
        # set seed
        setup_seed(self.T.seed)

    def on_eval(self):

        cmds = self.build_eval_cmd()
        if len(cmds) > 0:
            pool = Pool(processes=self.n_gpu)
            pool.map(run_process, cmds)

    def on_eval_end(self):
        if len(self.result_files) > 1:
            merge_metric(self.result_files, self.eval_tasks,
                         self.eval_model_path, self.D.llm_eval_name, self.logger)
        elif len(self.result_files) == 1:
            single_metric(self.result_files[0], self.eval_task,
                          self.eval_model_path, self.D.llm_eval_name, self.logger)

    def build_eval_cmd(self):
        if self.phase != "zst":
            pattern = os.path.join(self.T.checkpoint_file, "round-*")
            checkpoint_dirs = sorted(glob(pattern, recursive=True),
                                     key=lambda x: os.path.getctime(x), reverse=False)
            if len(checkpoint_dirs) == 0:  # single checkpoint
                checkpoint_dirs = [self.T.checkpoint_file]
            self.eval_model_path = self.T.checkpoint_file
        else:
            checkpoint_dirs = ["zst"]
            self.T.output_dir = os.path.join(self.T.output_dir, f"zst_{self.M.model_type}/")
            os.makedirs(self.T.output_dir, exist_ok=True)
            self.eval_model_path = self.T.output_dir

        gpu_list = self.T.eval_device.split(",")
        self.n_gpu = len(gpu_list)
        llm_eval_task = get_task(self.D.llm_eval_name)
        batch_size = "auto" if self.T.per_device_eval_batch_size == -1 else self.T.per_device_eval_batch_size
        max_batch_size = 64 if "auto" == batch_size else None

        cmds = []
        options = [
            "--model", "hf",
            "--batch_size", str(batch_size),
        ]
        if max_batch_size:
            options.extend(["--max_batch_size", f"{max_batch_size}"])
        options.extend(["--use_cache", f"{self.T.eval_reuse}"])

        cnt = 0
        self.result_files = []
        model_path = self.M.model_name_or_path
        for _, checkpoint_dir in enumerate(checkpoint_dirs):
            if checkpoint_dir == "zst":
                output_file_path = os.path.join(self.T.output_dir, f"{self.D.llm_eval_name}.json")
                model_args = f"pretrained={model_path},load_in_8bit=True,trust_remote_code=True"
            else:
                output_file_path = os.path.join(checkpoint_dir, f"{self.D.llm_eval_name}.json")
                model_args = f"pretrained={model_path},load_in_8bit=True,trust_remote_code=True,peft={checkpoint_dir}"

            self.result_files.append(output_file_path)
            if os.path.isfile(output_file_path):
                self.logger.info(f"result {output_file_path} exists, just skip it.")
                continue

            device = gpu_list[cnt % self.n_gpu]
            self.eval_tasks = ",".join(llm_eval_task) if len(llm_eval_task) > 1 else llm_eval_task[0]
            options.extend(
                [
                    "--model_args", model_args,
                    "--tasks", self.eval_tasks,
                    "--output_path", output_file_path,
                    "--device", f"cuda:{device}"
                ]
            )

            cmd = "lm_eval " + " ".join(options)
            cmds.append(cmd)
            cnt += 1

        self.logger.warning(f"run {len(cmds)} llm {self.D.llm_eval_name} tasks from {self.eval_model_path}")
        run_process("sleep 3s")

        return cmds
