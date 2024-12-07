import os
import re
import pickle
import copy
import json
import asyncio
from rouge import Rouge
# import openai


sys_prompt_dict = {
    "pairwise_default": 'Please act as an impartial judge and evaluate the quality of the outputs provided by two AI '
                        'assistants to the user question displayed below. Your evaluation should consider factors such '
                        'as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their '
                        'responses. Begin your evaluation by comparing the two outputs and provide a short '
                        'explanation. Avoid any position biases and ensure that the order in which the responses were '
                        'presented does not influence your decision. Do not allow the length of the responses to '
                        'influence your evaluation. Do not favor certain names of the assistants. Be as objective as '
                        'possible. ',
    "pairwise_score": 'Please act as an impartial judge and score the quality of the responses provided by two AI '
                      'assistants to the user Instruction displayed below. Your scores should includ four respects: '
                      'helpfulness (is it helpfulness to user’s Instruction), relevance (does it relevant to user’s '
                      'Instruction), correctness (does it contain correct knowledge), and coherence (is it fluently '
                      'and without grammar problems) and you give a score for each respect 1-5. Do not provide '
                      'explanation and copy any input from these respects. Avoid any position biases and ensure that '
                      'the order in which the responses were presented does not influence your decision. Do not allow '
                      'the length of the responses to influence your evaluation. Do not favor certain names of the '
                      'assistants. Be as objective as possible. '
}


def load_ref_labels(path):
    with open(path) as file:
        data = [json.loads(line) for line in file]
    print(len(data))
    return data


def build_compair_data(model_outputs, ref_outputs, ref_response_name):
    rouger = Rouge()

    def align_output(ref_output, model_output):
        scores = rouger.get_scores(model_output['instruction'], ref_output['instruction'])
        if scores[0]['rouge-l']['f'] > 0.9:
            return True
        else:
            return False

    for mp in model_outputs:
        for rfp in ref_outputs:
            if align_output(rfp, mp):
                mp['response'] = rfp[ref_response_name]
                break

    return model_outputs


def load_pred_labels(path):
    with open(path, "rb") as file:
        outputs = pickle.load(file)
    dataset = []
    for idx, instruct in enumerate(outputs['inputs']):
        instruct = instruct.split("## Instruction:")[1].split("### Response:")[0].replace("\n", "")
        dataset.append(
            {'response': outputs['labels'][idx],
             'output': outputs['preds'][idx],
             'instruction': instruct
             }
        )
    return dataset


def fmat_refer_model_name(engine):
    # 'gpt-4-answer', 'claude-2', 'gpt-3.5-turbo-answer', 'text-davinci-003-answer'
    if 'gpt4' in engine:
        return 'gpt-4-answer'
    elif 'claude2' in engine:
        return 'claude-2'
    elif 'gpt3.5' in engine:
        return 'gpt-3.5-turbo-answer'
    elif 'davinci003' in engine:
        return 'text-davinci-003-answer'
    elif 'llama2' in engine:
        return 'model_output'
    elif 'human' in engine:
        # return 'response'
        return 'output'
    else:
        raise ValueError(f"engine must in ['gpt4', 'claude2', 'gpt3.5', 'davinci003']")


def eval_make_prompt(template, val_dict):
    # text_to_format = re.findall("{([^ \s]+?)}", template)
    # prompt = copy.deepcopy(template)
    # for to_format in text_to_format:
    #     prompt = prompt.replace("{" + to_format + "}", val_dict[to_format], 1)
    formats = "\n## Instruction:\n{instruction}\n## Output a:\n{output_1}## Output b:\n{output_2}\n"
    formats = formats.format_map(val_dict)
    prompt = template + formats
    return prompt


def eval_encode_prompt(prompt, instruction, model_output, reference_output, args):
    """Encode multiple prompt instructions into a single string."""

    if args.reference_first:
        output_list = [reference_output, model_output]
    else:
        output_list = [model_output, reference_output]

    mapping_dict_output = {"instruction": instruction}
    mapping_dict_generator = {}
    for idx in range(2):
        mapping_dict_output['output_' + str(idx + 1)] = output_list[idx]['output']
        mapping_dict_generator['model_' + str(idx + 1)] = output_list[idx]['generator']

    filled_prompt = eval_make_prompt(prompt, mapping_dict_output)

    return filled_prompt, mapping_dict_generator


def load_or_convert_to_dataframe(dataset_path):
    if 'jsonl' in dataset_path:
        dataset = [json.loads(l) for l in open(dataset_path, "r")]
        # import pdb;pdb.set_trace()
    elif 'json' in dataset_path:
        with open(dataset_path, 'r') as file:
            dataset = json.load(file)
    else:
        raise ValueError("Unsupported file format. Please provide a .json or .jsonl file.")
    return dataset


def gpt_output_generation_encode_prompt(task_dict):
    """Encode multiple prompt instructions into a single string."""
    prompt = ""

    (instruction, task_input) = task_dict["instruction"], task_dict["input"]
    if instruction == "NA":
        return ""
    prompt += instruction + "\n\n"

    if task_input == "" or "<noinput>" in task_input or "NA" in task_input:
        return prompt
    prompt += task_input + "\n"
    return prompt
