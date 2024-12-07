import os
import pickle
import random
import numpy as np
import pandas as pd

import torch


"""
This module, `misc.py`, contains a collection of utility functions and data structures designed to support various tasks in a project. It includes methods for building prompts for legal tasks, exact matching for result comparison, and functionality to reuse results from previous runs by loading them from a file. The module is structured to facilitate tasks that require manipulation of text data, comparison of predicted results against actual results, and efficient handling of batch processing in scenarios where previous results can be leveraged to avoid redundant computations.

Key Components:
- `prompt_cases`: A dictionary mapping different prompt templates for legal case analysis.
- `legal_task_prompt`: A dictionary for storing specific prompt templates related to legal case prediction tasks.
- `build_legal_prompt`: Function to retrieve a specific legal task prompt template.
- `build_prompt`: Function to format a given prompt with provided text.
- `build_two_prompt`: Function to format a given prompt with provided text and add an additional step-by-step thinking prompt.
- `exact_matching`: Function to calculate the exact match ratio between predicted results and actual results.
- `reuse_results`: Function to load results from a file if available, and calculate the exact match ratio of these results, facilitating the reuse of previously computed predictions to save computational resources.

This module is designed to be a versatile tool in the processing and analysis of text data, especially in the context of legal case analysis, by providing efficient and reusable components.
"""


prompt_cases = {
    1: "现在你是一个法律专家，请你给出下面案件的类型。\n案件：{}\n案件类型是：",
    2: "现在你是一个法律专家，请你给出下面案件的类型。\n案件：{}\n案件类型：",
    3: "请给出下面法律案件的类型。\n案件：{}\n案件类型是：",
    4: "法律案件：{}\n该法律案件类型是：",
    5: "案件：{}\n案件类型是：",
    6: "现在你是一个法律专家，请你判断下面案件的类型。\n案件：{}\n案件类型是：",
    7: "现在你是一个法律专家，请你给出下面民事案件的具体类型。\n案件：{}\n案件类型是：",
    8: "现在你是一个法律专家，请你判断下面民事案件的具体类型。\n案件：{}\n案件类型是：",
    9: "现在你是一个法律专家，请你根据下面的法律分析方式给出下面民事案件的具体类型。法律分析方法：问题，规则，应用，结论。\n案件：{}\n案件类型是："
}

legal_task_prompt = {
    "lcp": "现在你是一个法律专家，请你判断下面案件的类型。\n案件：{}\n案件类型是："

}


def build_legal_prompt(task="lcp"):
    """
    Build a legal prompt based on the specified task.

    Args:
        task: The legal task for which the prompt is being generated.

    Returns:
        The formatted legal prompt based on the specified task

    """
    return legal_task_prompt[task]


def build_prompt(prompt, text):
    """
    Build a prompt by formatting the given text into the specified prompt template.

    Args:
        prompt: The prompt template to be used.
        text: The text to be inserted into the prompt template.

    Returns:
        The formatted prompt with the provided text.

    """
    return prompt.format(text)


def build_two_prompt(prompt, text):
    """
    Build a prompt by formatting the given text into the specified prompt template and adding a step-by-step
    thinking prompt.

    Args:
        prompt: The prompt template to be used.
        text: The text to be inserted into the prompt template.

    Returns:
        The formatted prompt with the provided text and a step-by-step thinking

    """
    cot_prompt = '让我们一步一步思考。'
    return prompt.format(text) + "\n{}".format(cot_prompt)


def exact_matching(pre_results, real_results):
    """
    Calculate the exact match ratio between predicted results and actual results.

    Args:
        pre_results: The predicted results.
        real_results: The actual results.

    Returns:
        The exact match ratio between the predicted and actual results.

    """
    cnt = 0
    correct = 0
    for pre_res, real_res in zip(pre_results, real_results):
        same_res = set(pre_res) & set(real_res)
        if len(same_res) == len(real_res):
            correct += 1
        cnt += 1
    return round(correct / cnt, 3)


def reuse_results(pth, results, batch_size=8):
    """
    Load results from a file if available, and calculate the exact match ratio of these results.

    Args:
        pth: The path to the file containing the results.
        results: The dictionary to store the results.
        batch_size: The batch size for processing the results.

    Returns:
        The number of batches to skip and the updated ``results`` dictionary.

    """

    skip_batch = -1
    if not os.path.isfile(pth):
        return skip_batch, results

    with open(pth, "rb") as file:
        old_results = pickle.load(file)
    
    # results {prompt:, result: [texts, labels, predict_labels]}
    label, predic = [], []
    for result in old_results["result"]:
        texts, labels, predict_labels = result
        label.append(labels)
        predic.append(predict_labels)
        results["result"].append((texts, labels, predict_labels))
    skip_batch = len(results["result"]) // batch_size
    reuse_num = skip_batch * batch_size
    results["result"] = results["result"][0:reuse_num]
    print(f"load {len(results['result'])} predictions, EM: {exact_matching(predic, label)}")

    return skip_batch, results
