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
import inspect
import json
import os
from dataclasses import asdict, dataclass, field
from typing import Dict, Optional, Union

from huggingface_hub import hf_hub_download
from transformers.utils import PushToHubMixin

from .utils import CONFIG_NAME, NeurocacheType, TaskType


@dataclass
class NeurocacheConfigMixin(PushToHubMixin):
    r"""
    This is the base configuration class for Neurocache. It inherits from
    [`~transformers.utils.PushToHubMixin`] which contains the methods to push your model to
    the Hub. The method `save_pretrained` will save the configuration of your adapter model
    in a directory. The method `from_pretrained` will load the configuration of your adapter
    model from a directory.
    Args:
        neurocache_type (`str`]): The type of Neurocache method to use.
    """
    neurocache_type: Union[str, NeurocacheType] = field(
        default=NeurocacheType.ONDEVICE,
        metadata={"help": "The type of Neurocache model."},
    )
    task_type: Union[str, TaskType] = field(
        default=TaskType.CAUSAL_LM, metadata={"help": "Task type"}
    )
    auto_mapping: Optional[dict] = field(
        default=None,
        metadata={"help": "An auto mapping dict to help retrieve the base model class if needed."},
    )

    def to_dict(self) -> Dict:
        return asdict(self)

    def save_pretrained(self, save_directory: str, **kwargs) -> None:
        r"""
        This method saves the configuration of your adapter model in a directory.

        Args:
            save_directory (`str`):
                The directory where the configuration will be saved.
            kwargs (additional keyword arguments, *optional*):
                Additional keyword arguments passed along to the
                [`~transformers.utils.PushToHubMixin.push_to_hub`] method.
        """
        if os.path.isfile(save_directory):
            raise AssertionError(
                f"Provided path ({save_directory}) should be a directory, not a file"
            )

        os.makedirs(save_directory, exist_ok=True)
        auto_mapping_dict = kwargs.pop("auto_mapping_dict", None)

        output_dict = asdict(self)
        # converting set type to list
        for key, value in output_dict.items():
            if isinstance(value, set):
                output_dict[key] = list(value)

        output_path = os.path.join(save_directory, CONFIG_NAME)

        # Add auto mapping details for custom models.
        if auto_mapping_dict is not None:
            output_dict["auto_mapping"] = auto_mapping_dict

        # save it
        with open(output_path, "w") as writer:
            writer.write(json.dumps(output_dict, indent=2, sort_keys=True))

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        subfolder: Optional[str] = None,
        **kwargs,
    ):
        r"""
        This method loads the configuration of your adapter model from a directory.

        Args:
            pretrained_model_name_or_path (`str`):
                The directory or the Hub repository id where the configuration is saved.
            kwargs (additional keyword arguments, *optional*):
                Additional keyword arguments passed along to the child class initialization.
        """
        path = (
            os.path.join(pretrained_model_name_or_path, subfolder)
            if subfolder is not None
            else pretrained_model_name_or_path
        )

        hf_hub_download_kwargs, class_kwargs, _ = cls._split_kwargs(kwargs)

        if os.path.isfile(os.path.join(path, CONFIG_NAME)):
            config_file = os.path.join(path, CONFIG_NAME)
        else:
            try:
                config_file = hf_hub_download(
                    pretrained_model_name_or_path,
                    CONFIG_NAME,
                    subfolder=subfolder,
                    **hf_hub_download_kwargs,
                )
            except Exception:
                raise ValueError(
                    f"Can't find '{CONFIG_NAME}' at '{pretrained_model_name_or_path}'"
                )

        loaded_attributes = cls.from_json_file(config_file)
        kwargs = {**class_kwargs, **loaded_attributes}
        config = cls(**kwargs)
        return config

    @classmethod
    def from_json_file(cls, path_json_file: str, **kwargs):
        r"""
        Loads a configuration file from a json file.

        Args:
            path_json_file (`str`):
                The path to the json file.
        """
        with open(path_json_file, "r") as file:
            json_object = json.load(file)

        return json_object

    @classmethod
    def _split_kwargs(cls, kwargs):
        hf_hub_download_kwargs = {}
        class_kwargs = {}
        other_kwargs = {}

        for key, value in kwargs.items():
            if key in inspect.signature(hf_hub_download).parameters:
                hf_hub_download_kwargs[key] = value
            elif key in list(cls.__annotations__):
                class_kwargs[key] = value
            else:
                other_kwargs[key] = value

        return hf_hub_download_kwargs, class_kwargs, other_kwargs


@dataclass
class NeurocacheConfig(NeurocacheConfigMixin):
    """
    The base configuration class to store the configuration of a [`NeurocacheModel`].

    Args:
        base_model_name_or_path (`str`, defaults to `None`):
            The name or path of the base model to use.
        inference_mode (`bool`, defaults to `False`): Whether to use the
            Neurocache model in inference mode.
        cache_size (`int`, defaults to 8192): The size of the cache (tokens).
        compression_factor (`int`, defaults to 8): The compression factor of the cache.
        global_cache (`bool`, defaults to `False`): Whether to use a global cache.
        similarity_fn (`str`, defaults to `l2`): The similarity function to use.
        topk (`int`, defaults to 32): The number of neighbors to retrieve.
        neighborhood_size (`int`, defaults to 2): Size of retrieved neighborhood.
        context_size (`int`, defaults to 2): Size of attention context window.
        cache_layers (`list`, defaults to `None`): Layers to store in cache.
            If `None`, the layer with idx == `num_layers * 3 // 4` is used.
        attention_layers (`list`, defaults to `None`): Layers to use for attention.
            If `None`, the layers with idx >= `num_layers * 3 // 4` are used.
    """

    base_model_name_or_path: str = field(
        default=None, metadata={"help": "The name of the base model to use."}
    )
    inference_mode: bool = field(default=False, metadata={"help": "Whether to use inference mode"})
    cache_size: int = field(default=8192, metadata={"help": "The size of the cache (tokens)"})
    compression_factor: int = field(
        default=8, metadata={"help": "The compression factor of the cache"}
    )
    global_cache: bool = field(default=False, metadata={"help": "Whether to use a global cache"})
    similarity_fn: str = field(default="l2", metadata={"help": "The similarity function to use"})
    topk: int = field(default=32, metadata={"help": "The number of neighbors to retrieve"})
    neighborhood_size: int = field(default=2, metadata={"help": "Size of retrieved neighborhood"})
    context_size: int = field(default=2, metadata={"help": "Size of attention context window"})
    cache_layers: list = field(default=None, metadata={"help": "Layers to store in cache"})
    attention_layers: list = field(default=None, metadata={"help": "Layers to use for attention"})
