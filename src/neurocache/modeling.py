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

from __future__ import annotations

import inspect
import os
from contextlib import contextmanager
from typing import Any, Dict, Optional

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import save_file as safe_save_file
from transformers import PreTrainedModel
from transformers.utils import PushToHubMixin

from .cache import OnDeviceCacheConfig
from .config import NeurocacheConfig
from .neurocache import Neurocache
from .utils import (
    SAFETENSORS_WEIGHTS_NAME,
    WEIGHTS_NAME,
    get_neurocache_model_state_dict,
    infer_device,
    load_neurocache_weights,
    set_neurocache_model_state_dict,
)


class NeurocacheModel(PushToHubMixin, torch.nn.Module):
    """
    Base model encompassing various Neurocache methods.

    Args:
        model ([`~transformers.PreTrainedModel`]): The base transformer model used for Neurocache.
        neurocache_config ([`NeurocacheConfig`]): The configuration of the Neurocache model.

    **Attributes**:
        - **base_model** ([`~transformers.PreTrainedModel`]) -- The base transformer
            model used for Neurocache.
        - **base_model_config** ([`~transformers.PretrainedConfig`]) -- The configuration
            of the base model.
        - **neurocache_config** ([`NeurocacheConfig`]) -- The configuration of
            the Neurocache model.
        - **base_cache** ([`Neurocache`]) -- The Neurocache module.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        neurocache_config: NeurocacheConfig,
    ):
        super().__init__()

        # initialize the base model and add the neurocache modules/hooks
        self.base_model = model
        self.base_model_config = model.config
        self.neurocache_config = neurocache_config
        self.base_cache = Neurocache(model, neurocache_config)

        self._mark_only_neurocache_as_trainable()

        # handle gradient checkpointing
        if getattr(model, "is_gradient_checkpointing", True):
            model = self._prepare_model_for_gradient_checkpointing(model)

    def reset_sequence_cache(self, start_of_sequence: torch.Tensor):
        r"""Resets the sequence cache. This method should be called at the
        beginning of each new sequence.

        Args:
            start_of_sequence (`torch.Tensor`):
                A tensor of shape `(batch_size,)` containing boolean values
                indicating whether the current segment is the start of a sequence.
        """
        self.base_cache.reset_cache(start_of_sequence)

    def reinitialize_cache(self):
        r"""Reinitializes the cache."""
        self.base_cache.reinitialize_cache()

    def save_pretrained(
        self,
        save_directory: str,
        safe_serialization: bool = False,
        **kwargs: Any,
    ):
        r"""
        This function saves the neurocache weights and the configuration files to a
        directory, so that it can be reloaded using the [`NeurocacheModel.from_pretrained`]
        class method, and also used by the [`NeurocacheModel.push_to_hub`] method.

        Args:
            save_directory (`str`):
                Directory where the weights and configuration files will
                be saved (will be created if it does not exist).
            kwargs (additional keyword arguments, *optional*):
                Additional keyword arguments passed along to the `push_to_hub` method.
        """
        if os.path.isfile(save_directory):
            raise ValueError(f"Provided path ({save_directory}) should be a directory, not a file")

        # save only the trainable weights
        output_state_dict = get_neurocache_model_state_dict(
            self,
            state_dict=kwargs.get("state_dict", None),
        )
        os.makedirs(save_directory, exist_ok=True)

        if safe_serialization:
            safe_save_file(
                output_state_dict,
                os.path.join(save_directory, SAFETENSORS_WEIGHTS_NAME),
                metadata={"format": "pt"},
            )
        else:
            torch.save(output_state_dict, os.path.join(save_directory, WEIGHTS_NAME))

        # save the config and change the inference mode to `True`
        if self.neurocache_config.base_model_name_or_path is None:
            self.neurocache_config.base_model_name_or_path = self.base_model.__dict__.get(
                "name_or_path", None
            )
        inference_mode = self.neurocache_config.inference_mode
        self.neurocache_config.inference_mode = True

        if self.neurocache_config.task_type is None:
            # deal with auto mapping
            base_model_class = self._get_base_model_class()
            parent_library = base_model_class.__module__

            auto_mapping_dict = {
                "base_model_class": base_model_class.__name__,
                "parent_library": parent_library,
            }
        else:
            auto_mapping_dict = None

        self.neurocache_config.save_pretrained(save_directory, auto_mapping_dict=auto_mapping_dict)
        self.neurocache_config.inference_mode = inference_mode

    @classmethod
    def from_pretrained(
        cls,
        model: PreTrainedModel,
        model_id: str,
        cache_type: str = "ONDEVICE",
        is_trainable: bool = False,
        config: Optional[NeurocacheConfig] = None,
        **kwargs: Any,
    ):
        if config is None:
            config = NEUROCACHE_TYPE_TO_CONFIG_MAPPING[cache_type].from_pretrained(
                model_id, **kwargs
            )

        if not isinstance(config, NeurocacheConfig):
            raise ValueError(
                f"The input config must be a NeurocacheConfig, got {config.__class__}"
            )

        config.inference_mode = not is_trainable
        task_type = (
            config.task_type
            if config.task_type in MODEL_TYPE_TO_NEUROCACHE_MODEL_MAPPING
            else "DEFAULT"
        )

        model = MODEL_TYPE_TO_NEUROCACHE_MODEL_MAPPING[task_type](model, config)
        model.load_neurocache_modules(model_id, is_trainable=is_trainable, **kwargs)
        return model

    def _prepare_model_for_gradient_checkpointing(self, model: PreTrainedModel):
        r"""
        Prepares the model for gradient checkpointing if necessary
        """
        if not (
            getattr(model, "is_loaded_in_8bit", False)
            or getattr(model, "is_loaded_in_4bit", False)
            or getattr(model, "is_quantized", False)
        ):
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            elif hasattr(model, "get_input_embeddings"):

                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)

                model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
        return model

    def get_nb_trainable_parameters(self):
        r"""
        Returns the number of trainable parameters and number of all parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for _, param in self.named_parameters():
            num_params = param.numel()
            # if using DS Zero 3 and the weights are initialized empty
            if num_params == 0 and hasattr(param, "ds_numel"):
                num_params = param.ds_numel

            # Due to the design of 4bit linear layers from bitsandbytes
            # one needs to multiply the number of parameters by 2 to get
            # the correct number of parameters
            if param.__class__.__name__ == "Params4bit":
                num_params = num_params * 2

            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params

        return trainable_params, all_param

    def print_trainable_parameters(self):
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params, all_param = self.get_nb_trainable_parameters()

        print(
            f"trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param}"
        )

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.base_model, name)

    def forward(self, *args: Any, start_of_sequence: torch.Tensor = None, **kwargs: Any):
        """
        Forward pass of the model.
        """
        if start_of_sequence is not None:
            self.reset_sequence_cache(start_of_sequence)
        return self.get_base_model()(*args, **kwargs)

    def _get_base_model_class(self):
        """
        Returns the base model class.
        """
        return self.base_model.__class__

    @contextmanager
    def disable_neurocache(self):
        """
        Disables neurocache.
        """
        try:
            self.base_cache._remove_hooks()
            yield
        finally:
            self.base_cache._register_hooks

    def get_base_model(self):
        """
        Returns the base model.
        """
        return self.base_model

    @classmethod
    def _split_kwargs(cls, kwargs: Dict[str, Any]):
        _kwargs_not_in_hf_hub_download_signature = ("use_auth_token",)
        hf_hub_download_kwargs = {}
        other_kwargs = {}

        for key, value in kwargs.items():
            if (
                key in inspect.signature(hf_hub_download).parameters
                or key in _kwargs_not_in_hf_hub_download_signature
            ):
                hf_hub_download_kwargs[key] = value
            else:
                other_kwargs[key] = value

        return hf_hub_download_kwargs, other_kwargs

    def load_neurocache_modules(self, model_id: str, is_trainable: bool = False, **kwargs: Any):
        hf_hub_download_kwargs, kwargs = self._split_kwargs(kwargs)
        torch_device = infer_device()

        neurocache_weights = load_neurocache_weights(
            model_id, device=torch_device, **hf_hub_download_kwargs
        )

        # load the weights into the model
        load_result = set_neurocache_model_state_dict(self, neurocache_weights)

        # Set model in evaluation mode to deactivate Dropout modules by default
        if not is_trainable:
            self.eval()
        return load_result

    @property
    def base_model_torch_dtype(self):
        return getattr(self.base_model, "dtype", None)

    def _mark_only_neurocache_as_trainable(self) -> None:
        for n, p in self.base_model.named_parameters():
            p.requires_grad = False

        for n, p in self.base_cache.named_parameters():
            p.requires_grad = True


class NeurocacheModelForCausalLM(NeurocacheModel):
    """
    Neurocache model for causal language modeling.

    Args:
        model ([`~transformers.PreTrainedModel`]): Base transformer model.
        neurocache_config ([`NeurocacheConfig`]): Neurocache config.

    Example:

        ```py
        >>> from transformers import AutoModelForCausalLM
        >>> from neurocache import NeurocacheModelForCausalLM
        >>> config = {
        ...     "neurocache_type": "ONDEVICE",
        ...     "task_type": "CAUSAL_LM",
        ...     "inference_mode": False,
        ...     "cache_size": 16384,
        ...     "compression_factor": 4,
        ...     "cache_layers": [30],
        ...     "attention_layers": [30, 31, 32, 33, 34, 35],
        ... }

        >>> neurocache_config = OnDeviceCacheConfig(**config)
        >>> model = AutoModelForCausalLM.from_pretrained("gpt2-large")
        >>> neurocache_model = NeurocacheModelForCausalLM(model, neurocache_config)
        >>> neurocache_model.print_trainable_parameters()
        trainable params: 1843200 || all params: 775873280 || trainable%: 0.23756456724479544
        ```
    """

    def __init__(self, model, neurocache_config: NeurocacheConfig):
        super().__init__(model, neurocache_config)
        self.base_model_prepare_inputs_for_generation = (
            self.base_model.prepare_inputs_for_generation
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        start_of_sequence=None,
        **kwargs,
    ):
        if start_of_sequence is not None:
            self.reset_sequence_cache(start_of_sequence)

        if self.base_model_config.model_type == "mpt":
            if inputs_embeds is not None:
                raise AssertionError("forward in MPTForCausalLM does not support inputs_embeds")
            return self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs,
            )

        return self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )

    def generate(self, start_of_sequence=None, **kwargs):
        if start_of_sequence is not None:
            self.reset_sequence_cache(start_of_sequence)

        self.base_model.prepare_inputs_for_generation = self.prepare_inputs_for_generation
        if hasattr(self.base_model, "model"):
            self.base_model.model.generation_config = self.generation_config
        else:
            self.base_model.generation_config = self.generation_config
        try:
            outputs = self.base_model.generate(**kwargs)
        except:
            self.base_model.prepare_inputs_for_generation = (
                self.base_model_prepare_inputs_for_generation
            )
            raise
        else:
            self.base_model.prepare_inputs_for_generation = (
                self.base_model_prepare_inputs_for_generation
            )
            return outputs

    def prepare_inputs_for_generation(self, *args, task_ids: torch.Tensor = None, **kwargs):
        model_kwargs = self.base_model_prepare_inputs_for_generation(*args, **kwargs)
        return model_kwargs


MODEL_TYPE_TO_NEUROCACHE_MODEL_MAPPING = {
    "DEFAULT": NeurocacheModel,
    "CAUSAL_LM": NeurocacheModelForCausalLM,
}

NEUROCACHE_TYPE_TO_CONFIG_MAPPING = {
    "ONDEVICE": OnDeviceCacheConfig,
}
