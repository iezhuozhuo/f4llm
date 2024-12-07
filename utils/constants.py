"""This module, `constants.py`, defines various constants used throughout the FNLP (Federated Natural Language
Processing) project. It includes constants for parameter-efficient tuning types, legal case prompts, and mappings
between client IDs and prompt types for legal case prediction. These constants are essential for the project's legal
case prediction tasks, enabling dynamic prompt selection based on the context and ensuring consistent use of tuning
techniques across the project.

Key Components:
- `petuning_type`: A list of parameter-efficient tuning types such as LoRA, Adapter, BitFit, Prefix, and P-Tuning.

"""


petuning_type = ["lora", "adapter", "bitfit", "prefix", "p-tuning"]
