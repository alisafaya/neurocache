from __future__ import annotations

import functools

import torch
from torch import nn
from transformers import PretrainedConfig, PreTrainedModel

from neurocache.utils import COMMON_MODULE_NAMES, NeurocacheType

from .cache import BatchedCache, OnDeviceCache
from .config import NeurocacheConfig


Array = torch.Tensor


def get_attribute(object, attr_name):
    for option in COMMON_MODULE_NAMES[attr_name]:
        if hasattr(object, option):
            return getattr(object, option)
    raise ValueError("Cannot infer attribute. Options: {}".format(attr_name))


class CacheAttention(nn.Module):
    """
    Wrapper that adds cache attention to a layer.

    Decoder Layer: (
        ...
        self_attn
        cache_attn
        layer_norm
        ...
    )

    Args:
        config: Configuration object containing the parameters of model.
        base_layer: Layer to be wrapped.
    """

    def __init__(
        self,
        base_decoder_layer: nn.Module,
        config: NeurocacheConfig,
        base_config: PretrainedConfig,
    ):
        super().__init__()
        self.config = config

        self.hidden_size = get_attribute(base_config, "hidden_size")
        self.num_heads = get_attribute(base_config, "num_heads")

        self.head_dim = self.hidden_size // self.num_heads
        self.scaler = self.head_dim**-0.5
        self.r_hidden_size = self.hidden_size // self.config.compression_factor

        # TODO: add support for GQA (for big llamas and mistral)
        # TODO: add bias option based on the model config
        self.k_proj = nn.Linear(self.r_hidden_size, self.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.r_hidden_size, self.num_heads * self.head_dim, bias=False)
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.init_proj_weights(get_attribute(base_decoder_layer, "self_attn"))

    def init_proj_weights(self, self_attn_module):
        """Initialize projection weights with self attention weights."""
        self.q_proj.weight.data = get_attribute(self_attn_module, "q_proj").weight.data.clone()
        self.o_proj.weight.data = get_attribute(self_attn_module, "o_proj").weight.data.clone()

        if self.config.compression_factor == 1:
            # copy the weights
            self.k_proj.weight.data = get_attribute(self_attn_module, "k_proj").weight.data.clone()
            self.v_proj.weight.data = get_attribute(self_attn_module, "v_proj").weight.data.clone()
        else:
            # slice the weights
            self.k_proj.weight.data = (
                get_attribute(self_attn_module, "k_proj").weight[:, : self.r_hidden_size].clone()
            )
            self.v_proj.weight.data = (
                get_attribute(self_attn_module, "v_proj").weight[:, : self.r_hidden_size].clone()
            )

    def ext_attention(
        self,
        external_keys: Array,
        external_values: Array,
        queries: Array,
    ) -> Array:
        """Attention over (keys, values) retrieved from cache.

        Args:
            external_keys: per-query keys from cache, of shape
                [batch_size, num_queries, num_heads, num_neighbors, head_size]
            external_values: per-query values from cache, of shape
                [batch_size, num_queries, num_heads, num_neighbors, head_size]
            queries: current queries, of shape:
                [batch_size, num_queries, num_heads, head_size]

        Returns:
            Attention outputs of shape [batch_size, num_queries, num_heads, head_size]
        """
        assert external_keys.ndim == 5
        assert queries.ndim == 4
        assert external_values.shape == external_keys.shape

        # Compute attention weights.
        ext_attn = torch.einsum("...qhd,...qhid->...hqi", queries, external_keys)
        ext_attn = nn.functional.softmax(ext_attn, dim=-1, dtype=torch.float32).to(queries.dtype)

        # Compute weighted sum of values.
        attn_output = torch.einsum("...hqi,...qhid->...qhd", ext_attn, external_values)
        return attn_output

    def forward(self, hidden_states, ext_hidden_states):
        """Attention over retrieved cached states.
            1. Project hidden states into queries.
            2. Project cached states into keys and values.
            3. Repeat neighbors for each query if context size > 1.
            4. Compute attention over keys and values.
            5. Project attention outputs back to hidden states.
            6. Residual connection with hidden states.

        Args:
            hidden_states: output of the previous layer.
                [batch_size, seq_len, hidden_size]
            ext_hidden_states: hidden states retrieved from cache.
                [batch_size, seq_len, num_neighbor, hidden_size]
        """
        assert ext_hidden_states.shape[-1] == self.r_hidden_size
        assert hidden_states.shape[-1] == self.hidden_size
        batch_size, seq_len, hidden_size = hidden_states.shape
        _, _, num_neighbors, r_hidden_size = ext_hidden_states.shape

        # Project hidden states into queries.
        queries = self.q_proj(hidden_states) * self.scaler
        queries = queries.view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Project external states into keys and values.
        keys = self.k_proj(ext_hidden_states)
        keys = keys.view(batch_size, seq_len, self.num_heads, num_neighbors, self.head_dim)
        values = self.v_proj(ext_hidden_states)
        values = values.view(batch_size, seq_len, self.num_heads, num_neighbors, self.head_dim)

        def repeat_neighbors(neighbors: Array, ctx_size: int):
            assert neighbors.ndim == 5
            # Allows access to previous tokens neighbors
            # i-th token will have the neighbors of all tokens from (i - c) to i.
            # We clip so the first tokens do not have negative indices
            context = (
                torch.arange(seq_len)[:, None] + torch.arange(-ctx_size + 1, 1)[None, :]
            ).clamp_(0, seq_len - 1)

            neighbors = neighbors.moveaxis(1, 2)
            neighbors = neighbors[:, :, context]
            neighbors = neighbors.reshape(batch_size, self.num_heads, seq_len, -1, self.head_dim)
            neighbors = neighbors.moveaxis(1, 2)
            return neighbors

        # Repeat neighbors for each query.
        if self.config.context_size > 1:
            keys = repeat_neighbors(keys, self.config.context_size)
            values = repeat_neighbors(values, self.config.context_size)

        # Compute attention over keys and values.
        attn_output = self.ext_attention(keys, values, queries)

        # Project attention outputs back to hidden states.
        attn_output = attn_output.view(batch_size, seq_len, hidden_size)
        attn_output = self.o_proj(attn_output)

        # Residual connection with hidden states.
        attn_output = attn_output + hidden_states
        return attn_output


class Neurocache(nn.Module):
    """
    Wrapper that handles cache retrieval, attention updates.

    Args:
        config: Configuration object containing the parameters of model.
        base_layer: Layer to be wrapped.
    """

    def __init__(
        self,
        base_model: PreTrainedModel,
        config: NeurocacheConfig,
    ):
        super().__init__()
        self.config = config
        self.base_model_config = base_model.config
        self.hidden_size = get_attribute(base_model.config, "hidden_size")

        if self.config.compression_factor > 1:
            # project hidden states to lower dimension and normalize
            def _weight_init(m):
                if isinstance(m, nn.Linear):
                    nn.init.trunc_normal_(m.weight, mean=0.0, std=m.in_features**-0.5)

            self.cache_dim = self.hidden_size // self.config.compression_factor
            self.h_proj = nn.Linear(self.hidden_size, self.cache_dim, bias=False)
            self.h_proj.apply(_weight_init)
            self.h_norm = nn.LayerNorm(self.cache_dim, eps=1e-6)
        else:
            # no-op
            self.h_proj = nn.Identity()
            self.h_norm = nn.Identity()
            self.cache_dim = self.hidden_size

        self.retrieval_state = {}
        self.cache_attns = nn.ModuleList([])
        self.caches = nn.ModuleList([])
        self.hooks = self._register_hooks(base_model)
        self.enabled = True

    def enable(self):
        """Enable cache."""
        self.enabled = True

    def disable(self):
        """Disable cache."""
        self.enabled = False

    def _remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.caches = nn.ModuleList([])
        self.cache_attns = nn.ModuleList([])
        self.hooks = []

    def reinitialize_cache(self):
        """Reinitialize cache.
        This can be used to change the batch size of the cache.
        """
        self.retrieval_state = {}
        for cache in self.caches:
            cache.wrapped = None

    def _register_hooks(self, base_model: PreTrainedModel):
        # Infer decoder layer list
        if hasattr(base_model, "model"):
            base_model = base_model.model

        if hasattr(base_model, "decoder"):
            base_model = base_model.decoder

        layers = get_attribute(base_model, "layers")
        num_layers = len(layers)

        if self.config.cache_layers is None or len(self.config.cache_layers) == 0:
            cache_layers = [
                num_layers * 3 // 4,
            ]
        else:
            cache_layers = self.config.cache_layers

        if self.config.attention_layers is None or len(self.config.attention_layers) == 0:
            attention_layers = range(cache_layers, num_layers)
        else:
            attention_layers = self.config.attention_layers

        # Check that cache layers are before attention layers
        assert min(attention_layers) >= min(cache_layers)

        # Register hooks and initialize cache attention layers
        hooks = []
        for i, idx in enumerate(sorted(cache_layers)):
            split_dims = () if self.config.global_cache else (0,)
            self.caches.append(BatchedCache(None, split_dims))
            hook = get_attribute(layers[idx], "self_attn").register_forward_hook(
                functools.partial(self.retrieve_hook, idx=i), with_kwargs=True
            )
            hooks.append(hook)

        for i, idx in enumerate(sorted(attention_layers)):
            self.cache_attns.append(
                CacheAttention(layers[idx], self.config, self.base_model_config)
            )
            hook = get_attribute(layers[idx], "self_attn").register_forward_hook(
                functools.partial(self.attention_hook, idx=i), with_kwargs=True
            )
            hooks.append(hook)

        return hooks

    def reset_cache(self, start_of_sequence: Array):
        """Reset cache at the beginning of a sequence."""
        if not self.config.global_cache:
            for cache in self.caches:
                if cache.wrapped is not None:
                    cache.reset(start_of_sequence)

    def attention_hook(self, module, args, kwargs, outputs, idx):
        """External attention hook."""
        if not self.enabled:
            return

        hs = kwargs["hidden_states"]
        ext_hs = self.retrieval_state["ext_hidden_states"]
        outputs = (self.cache_attns[idx](hs, ext_hs),) + outputs[1:]
        return outputs

    def retrieve_hook(self, module, args, kwargs, outputs, idx):
        """Hook to retrieve the hidden states of a layer."""
        if not self.enabled:
            return

        hs = kwargs["hidden_states"].detach()

        # TODO: Infer padding mask from attention mask
        input_mask = None

        # Project hidden states to lower dimension and normalize
        with torch.no_grad():
            phs = self.h_norm(self.h_proj(hs))

        # 1. Retrieve topk neighbors from cache

        # Since we need to infer the batch size before initializing
        # the cache we initialize the cache in the first forward pass
        # and retrieve in the second forward pass
        if (
            self.caches[idx].wrapped is None
            and self.config.neurocache_type == NeurocacheType.ONDEVICE
        ):
            batch_size = hs.shape[0]
            value_dim = hs.shape[-1] if (self.training and not self.config.inference_mode) else 0
            self.caches[idx].wrapped = OnDeviceCache(
                batch_size,
                self.config.cache_size,
                self.cache_dim,
                value_dim,
                self.config.neighborhood_size,
                self.config.similarity_fn,
                dtype=phs.dtype,
                ordering=self.config.cache_type,
            )

            # return zeros since cache is empty
            ext_hs = torch.zeros(
                batch_size,
                hs.shape[1],
                self.config.topk,
                self.cache_dim,
                dtype=hs.dtype,
            )

        else:
            # retrieve topk neighbors from cache
            assert self.caches[idx].wrapped is not None
            assert (
                self.caches[idx].wrapped.num_caches == hs.shape[0] or self.config.global_cache
            ), (
                "Cache batch size does not match hidden states batch size. "
                "Please re-initalize Neurocache."
            )

            keys, values = self.caches[idx].topk_retrieval(phs, input_mask, self.config.topk)

            if self.training and not self.config.inference_mode:
                # re-project retrieved values to train h_proj and h_norm layers
                ext_hs = self.h_norm(self.h_proj(values))
            else:
                ext_hs = keys

        # Store retrieved states for attention in following layers
        self.retrieval_state["ext_hidden_states"] = ext_hs

        # 2. Update cache with new hidden states

        # If training keep both projected and original hidden states
        # to train the h_proj and h_norm layers, otherwise keep only
        # projected hidden states
        if self.config.inference_mode or not self.training:
            self.caches[idx].update(phs, None, input_mask)
        elif self.config.neurocache_type == NeurocacheType.ONDEVICE:
            self.caches[idx].update(phs, hs, input_mask)
