import torch
from accelerate.utils import is_npu_available, is_xpu_available
from huggingface_hub import get_hf_file_metadata, hf_hub_url
from huggingface_hub.utils import EntryNotFoundError


# Get current device name based on available devices
def infer_device():
    if torch.cuda.is_available():
        torch_device = "cuda"
    elif is_xpu_available():
        torch_device = "xpu"
    elif is_npu_available():
        torch_device = "npu"
    else:
        torch_device = "cpu"
    return torch_device


def hub_file_exists(
    repo_id: str, filename: str, revision: str = None, repo_type: str = None
) -> bool:
    r"""
    Checks if a file exists in a remote Hub repository.
    """
    url = hf_hub_url(repo_id=repo_id, filename=filename, repo_type=repo_type, revision=revision)
    try:
        get_hf_file_metadata(url)
        return True
    except EntryNotFoundError:
        return False


NEUROCACHE_SUPPORTED_MODELS = [
    "opt",
    "llama",
    "mistral",
    "gptj",
]

COMMON_MODULE_NAMES = {
    "num_layers": ["num_hidden_layers", "num_layers", "n_layers"],
    "num_heads": ["num_attention_heads", "num_heads", "n_heads"],
    "hidden_size": ["hidden_size", "d_model", "embed_dim"],
    "self_attn": ["self_attention", "self_attn", "attn", "dec_attn"],
    "o_proj": ["o_proj", "out_proj", "output_proj", "output_projection"],
    "q_proj": ["q_proj", "query_proj", "query_projection"],
    "k_proj": ["k_proj", "key_proj", "key_projection"],
    "v_proj": ["v_proj", "value_proj", "value_projection"],
    "layers": ["layers", "h", "block", "blocks", "layer"],
}

WEIGHTS_NAME = "neurocache_model.bin"
SAFETENSORS_WEIGHTS_NAME = "neurocache_model.safetensors"
CONFIG_NAME = "neurocache_config.json"
