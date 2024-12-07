def get_delta_key(delta_type):
    """
    Get the delta key for the given delta type

    Args:
        delta_type (str): The delta type

    Returns:
        str: The delta key
    """
    delta_keys = {
        "fine-tuning": "",
        "prefix": "num_virtual_tokens",
        "bitfit": "",
        "lora": "lora_rank",
        "adapter": "bottleneck_dim",
        "emulator": ""
    }
    delta_keys_abb = {
        "fine-tuning": "ft",
        "emulator": "",
        "prefix": "ptn",
        "bitfit": "",
        "lora": "la",
        "adapter": "dim"
    }
    return delta_keys[delta_type], delta_keys_abb[delta_type]
