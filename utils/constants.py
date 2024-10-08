"""
This module, `constants.py`, defines various constants used throughout the FNLP (Federated Natural Language Processing) project. It includes constants for parameter-efficient tuning types, legal case prompts, and mappings between client IDs and prompt types for legal case prediction. These constants are essential for the project's legal case prediction tasks, enabling dynamic prompt selection based on the context and ensuring consistent use of tuning techniques across the project.

Key Components:
- `petuning_type`: A list of parameter-efficient tuning types such as LoRA, Adapter, BitFit, Prefix, and P-Tuning.
- `lcp_prompts`: A dictionary mapping prompt IDs to their corresponding text templates for legal case prediction.
- `lcp_client2prompt`: A nested dictionary mapping client IDs to prompt IDs, facilitating customized prompt selection for different clients.

These constants play a crucial role in the FNLP project by providing a centralized repository of configurations for legal case prediction, thereby enhancing modularity, maintainability, and scalability of the project's codebase.
"""


petuning_type = ["lora", "adapter", "bitfit", "prefix", "p-tuning"]

lcp_prompts = {
    # general
    0: "[Round 1]\n\n问：{}\n\n答：",
    # role play 0.180
    8: "现在你是一个法律专家，请你判断下面民事案件的具体类型。\n案件：{}\n案件类型是：",
    # zero-shot 0.137
    4: "法律案件：{}\n该法律案件类型是：",
    # few-shot-general
    2: "请给出案件的类型。例如案件：原告闫汉升向本院提出诉讼请求：1.判令被告向原告支付拉沙款4500元；2.本案诉讼费由被告承担。事实和理由：2017年4月4日，原告为被告所开办的石料厂拉沙子、石子3车共计产生费用4500"
       "元。当时被告称自己资金紧张过段时间结清运费，后经原告多次催要无果。\n案件类型是：运输合同纠纷。案件：{}\n案件类型是：",

    # zero-shot 0.082
    3: "请给出下面法律案件的类型。\n案件：{}\n案件类型是：",
    # zs 0.072
    5: "案件：{}\n案件类型是：",
    6: "现在你是一个法律专家，请你判断下面案件的类型。\n案件：{}\n案件类型是：",
    7: "现在你是一个法律专家，请你给出下面民事案件的具体类型。\n案件：{}\n案件类型是：",
    # zs
    1: "现在你是一个法律专家，请你给出下面案件的类型。\n案件：{}\n案件类型是：",
    9: "案件：{}\n该民事案件的具体类型是：",
    # cot
    21: "现在你是一个法律专家，请你根据下面的法律分析方式给出下面民事案件的具体类型。法律分析方法：问题，规则，应用，结论。\n案件：{}\n案件类型是：",
}

lcp_client2prompt = {
    42: {
        19: 0, 5: 0, 14: 0, 4: 0, 9: 0, 20: 0,
        13: 2, 15: 2, 18: 2, 6: 2, 12: 2,
        17: 4, 10: 4, 1: 4, 11: 4, 2: 4,
        16: 8, 7: 8, 8: 8, 0: 8, 3: 8},
    3407: {
        15: 0, 11: 0, 17: 0, 0: 0, 20: 0, 1: 0,
        2: 2, 19: 2, 14: 2, 16: 2, 8: 2,
        7: 4, 3: 4, 5: 4, 9: 4, 18: 4,
        12: 8, 6: 8, 10: 8, 4: 8, 13: 8},
    10: {
        3: 0, 11: 0, 5: 0, 17: 0, 19: 0, 18: 0,
        8: 2, 9: 2, 12: 2, 16: 2, 2: 2,
        10: 4, 4: 4, 14: 4, 20: 4, 7: 4,
        6: 8, 0: 8, 15: 8, 13: 8, 1: 8},
    0: {
        10: 0, 19: 0, 17: 0, 14: 0, 0: 0,
        12: 0, 18: 2, 11: 2, 2: 2, 3: 2, 9: 2,
        5: 4, 7: 4, 4: 4, 20: 4, 6: 4,
        15: 8, 16: 8, 8: 8, 1: 8, 13: 8},
    -1: {
        0: 0, 1: 0, 2: 0, 3: 4, 4: 0,
        5: 0, 6: 4, 7: 2, 8: 4, 9: 4,
        10: 0, 11: 4, 12: 0, 13: 4, 14: 8,
        15: 4, 16: 4, 17: 4, 18: 4,
        19: 8, 20: 8
    }
}
