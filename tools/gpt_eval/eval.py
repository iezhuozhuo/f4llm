import json
import os
import sys
import time
import random
import argparse
from glob import glob
# import openai
from tqdm import tqdm
from loguru import logger

sys.path.append('..')  # Add the parent directory to sys.path
from utils import *

random.seed(42)


# openai.api_key = os.getenv("OPENAI_API_KEY")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_output",
        type=str,
        required=True,
        default=None,
        help="The path to the model output data.",
    )
    parser.add_argument(
        "--reference_output",
        type=str,
        default=None,
        help="The path to the reference output data.",
    )
    parser.add_argument(
        "--output_file_name",
        type=str,
        required=False,
        default=None,
        help="The file name for output.",
    )
    parser.add_argument(
        "--refer_model_name",
        type=str,
        required=False,
        default="gpt-4",
        help="The reference model name.",
        # 'gpt-4-answer', 'claude-2', 'gpt-3.5-turbo-answer', 'text-davinci-003-answer'
    )
    parser.add_argument(
        "--engine",
        type=str,
        default="gpt4",
        help="The api engine to compare.",
        # 'gpt4', 'claude2', 'gpt3.5', 'davinci003'
    )
    parser.add_argument(
        "--request_batch_size",
        type=int,
        default=5,
        help="The number of requests to send to GPT3 at a time."
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=20,
        help="Max input tokens."
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=5,
        help="failed retry times."
    )
    parser.add_argument(
        "--reference_first",
        action="store_true",
        help="If pass reference model will be model_1, otherwise reference model will be model_2.",
    )

    parser.add_argument(
        "--task_name",
        type=str,
        default=None,
        help="The task_name for different instruction usage."
    )

    parser.add_argument(
        "--max_test_number",
        type=int,
        default=-1,
        help="set if the test instances is less than generation.",
    )
    parser.add_argument(
        "--show_case",
        type=bool,
        default=False,
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logger.info(f"Eval Start, {args.engine} as reference model")

    if os.path.isfile(args.model_output):
        model_output_files = [args.model_output]
        args.batch_dir = os.path.dirname(args.model_output)
    elif os.path.isdir(args.model_output):
        pattern = os.path.join(args.model_output, "*/*.pkl")
        model_output_files = sorted(glob(pattern, recursive=True),
                                    key=lambda x: os.path.getctime(x), reverse=False)
        args.batch_dir = args.model_output
    else:
        raise ValueError

    algs = ""
    if args.task_name in args.model_output:
        all_list = args.model_output.split("/")
        out_idx = all_list.index(args.task_name)
        algs = all_list[out_idx+1]

    # ref_data = load_ref_labels(args.reference_output)
    if args.reference_output is None:
        if args.engine in ["gpt4", "gpt3.5", "human"]:
            root_folder = os.path.dirname(os.path.abspath(__file__))
            args.reference_output = os.path.join(root_folder,
                                                 f"../../../../data/fed{args.task_name}/"
                                                 f"{args.task_name}-test-{args.engine}.json"
                                                 )
        else:
            raise ValueError

    ref_data = load_or_convert_to_dataframe(args.reference_output)
    system_prompt = sys_prompt_dict['pairwise_score']
    prompt = open("./tools/prompts/pairwise_scorev1_1.txt").read() + "\n"

    for mop in model_output_files:
        args.model_output = mop
        logger.info(f"processing {args.model_output}")
        dataset = load_pred_labels(args.model_output)
        if args.max_test_number != -1:
            dataset = dataset[:args.max_test_number]

        # if args.task_name in ['medsi', "alpt", "medalp", ""]:
        #     instructions = [item['instruction'] + "\n\n" for item in dataset]
        # else:
        #     raise ValueError("Unsupported task.")
        instructions = [item['instruction'] + "\n\n" for item in dataset]

        args.refer_model_name = fmat_refer_model_name(args.engine)
        dataset = build_compair_data(dataset, ref_data, args.refer_model_name)

        model_output = [{"generator": "model", "output": item['output']} for item in dataset]
        reference_output = [{"generator": args.engine, "output": item['response']} for item in dataset]

        assert len(model_output) == len(reference_output) == len(instructions)
        total = len(reference_output)

        results = []

        # save_data path
        file_name = args.model_output.split("/")[-1].split(".")[0]
        if "zst" in args.model_output:
            model_details = ""
        else:
            try:
                model_details = args.model_output.split("/")[-2].split("_")[1] + "_"
            except:
                model_details = ""
        file_name = model_details + file_name + "_"
        output_file_name = algs + "_" + file_name + args.engine + ".json"
        output_path = os.path.join(args.batch_dir, output_file_name)

        message_list = []
        for idx in range(total):
            instr, m_o, r_o = instructions[idx], model_output[idx], reference_output[idx]

            for reference_first in [True, False]:
                args.reference_first = reference_first
                flag = "first" if args.reference_first else "last"
                task_prompt, _ = eval_encode_prompt(prompt, instr, m_o, r_o, args)
                message = {"idx": f"{idx}-{flag}",
                           "system": system_prompt,
                           "question": task_prompt
                           }
                message_list.append(message)
        if args.show_case:
            logger.info(f"case study:\n {message_list[0]}")
            logger.info(f"case study:\n {message_list[1]}")
            args.show_case = False

        with open(output_path, "w") as file:
            json.dump(message_list, file)
        logger.info(f"save tst {len(message_list)} cases in {output_path}")
