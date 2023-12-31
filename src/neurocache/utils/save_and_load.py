# coding=utf-8
# Adapted from the PEFT library: https://github.com/huggingface/peft/
# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
from typing import Optional

import torch
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError
from peft import get_peft_model_state_dict, set_peft_model_state_dict
from safetensors.torch import load_file as safe_load_file

from .types import NeurocacheType
from .utils import SAFETENSORS_WEIGHTS_NAME, WEIGHTS_NAME, hub_file_exists, infer_device


def get_neurocache_model_state_dict(model, unwrap_compiled=False):
    """
    Get the state dict of the Neurocache model.

    Args:
        model ([`NeurocacheModel`]): The Neurocache model.
            When using torch.nn.DistributedDataParallel, DeepSpeed or FSDP, the model
            should be the underlying model/unwrapped model (i.e. model.module).
        state_dict (`dict`, *optional*, defaults to `None`):
            The state dict of the model. If not provided, the state dict of the passed
            model will be used.
    """
    if unwrap_compiled:
        model = getattr(model, "_orig_mod", model)

    # Get the state dict of neurocache modules
    state_dict = model.base_cache.state_dict()
    # Get the state dict of the adapters
    try:
        if hasattr(model.base_model, "peft_config"):
            state_dict.update(
                get_peft_model_state_dict(model.base_model, adapter_name="neurocache")
            )
    except KeyError:
        pass
    return state_dict


def set_neurocache_model_state_dict(model, neurocache_model_state_dict):
    """
    Set the state dict of the Neurocache model.

    Args:
        model ([`NeurocacheModel`]): The Neurocache model.
        neurocache_model_state_dict (`dict`): The state dict of the model.
    """
    config = model.neurocache_config

    if config.neurocache_type == NeurocacheType.ONDEVICE:
        load_result = model.base_cache.load_state_dict(neurocache_model_state_dict, strict=False)
        try:
            if hasattr(model.base_model, "peft_config"):
                set_peft_model_state_dict(
                    model.base_model, neurocache_model_state_dict, adapter_name="neurocache"
                )
        except KeyError:
            pass
    else:
        raise NotImplementedError

    return load_result


def load_neurocache_weights(
    model_id: str, device: Optional[str] = None, **hf_hub_download_kwargs
) -> dict:
    r"""
    A helper method to load the weights from the HuggingFace Hub or locally

    Args:
        model_id (`str`):
            The local path to the weights or the name to load from the HuggingFace Hub.
        device (`str`):
            The device to load the weights onto.
        hf_hub_download_kwargs (`dict`):
            Additional arguments to pass to the `hf_hub_download` method when loading
            from the HuggingFace Hub.
    """
    path = (
        os.path.join(model_id, hf_hub_download_kwargs["subfolder"])
        if hf_hub_download_kwargs.get("subfolder", None) is not None
        else model_id
    )

    if device is None:
        device = infer_device()

    if os.path.exists(os.path.join(path, SAFETENSORS_WEIGHTS_NAME)):
        filename = os.path.join(path, SAFETENSORS_WEIGHTS_NAME)
        use_safetensors = True
    elif os.path.exists(os.path.join(path, WEIGHTS_NAME)):
        filename = os.path.join(path, WEIGHTS_NAME)
        use_safetensors = False
    else:
        has_remote_safetensors_file = hub_file_exists(
            model_id,
            SAFETENSORS_WEIGHTS_NAME,
            revision=hf_hub_download_kwargs.get("revision", None),
            repo_type=hf_hub_download_kwargs.get("repo_type", None),
        )
        use_safetensors = has_remote_safetensors_file

        if has_remote_safetensors_file:
            # Priority 1: load safetensors weights
            filename = hf_hub_download(
                model_id,
                SAFETENSORS_WEIGHTS_NAME,
                **hf_hub_download_kwargs,
            )
        else:
            try:
                filename = hf_hub_download(model_id, WEIGHTS_NAME, **hf_hub_download_kwargs)
            except EntryNotFoundError:
                raise ValueError(
                    f"Can't find weights for {model_id} in {model_id} or in the Hugging Face Hub. "
                    f"Please check that the file {WEIGHTS_NAME} or {SAFETENSORS_WEIGHTS_NAME} "
                    f"is present at {model_id}."
                )

    if use_safetensors:
        weights = safe_load_file(filename, device=device)
    else:
        weights = torch.load(filename, map_location=torch.device(device))

    return weights
