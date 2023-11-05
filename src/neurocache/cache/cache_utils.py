import abc
import enum
from typing import Callable, Sequence, Tuple, TypeVar, Union

import numpy as np
import torch
from torch import nn


Shape = Sequence[int]
Dtype = torch.dtype
Array = torch.Tensor

Axes = Union[int, Tuple[int, ...]]
F = TypeVar("F", bound=Callable)


class CacheType(str, enum.Enum):
    FIFO = "FIFO"
    LRU = "LRU"


class Cache(nn.Module, metaclass=abc.ABCMeta):
    """Internal interface for cache layers without batch dim.

    See BatchedCache for a layer that can be used in PyTorch models.
    """

    def __init__(self, num_caches: int):
        super().__init__()
        self.num_caches = num_caches

    @abc.abstractmethod
    def update(self, key: Array, value: Array, mask: Array) -> int:
        """Adds key/value pairs to cache.
        Args:
            key: of shape (num_kv, num_caches, k_features)
            value: of shape (num_kv, num_caches, v_features)
            mask: of shape (num_kv, num_caches, 1) indicating which keys should be
                updated. If None, all keys are updated.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def topk_retrieval(self, query: Array, mask: Array, num_neighbors: int) -> Tuple[Array, Array]:
        """Retrieves the nearest neighbors for each query.

        Args:
            query: of shape (num_queries, num_caches, k_features)
            num_neighbors: int indicating the number of neighbors to retrieve

        Returns:
            Tuple of selected keys and selected values of shapes
            (num_queries, num_caches, num_neighbors, k_features), and
            (num_queries, num_caches, num_neighbors, v_features)
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def reset(self, caches: Array) -> int:
        """Reset some or all of the caches in the cache.

        Args:
            caches: A vector of shape (num_caches) of type bool. Each position
                indicates whether the cache with the same index should be reset.
        """
        raise NotImplementedError()

    def forward(self, query, num_neighbors):
        return self.topk_retrieval(query, num_neighbors)


def _target_dimensions(shape: Shape, source_dimensions: Sequence[int]) -> Sequence[int]:
    target_dimensions = range(-2, -2 - len(source_dimensions), -1)
    assert len(source_dimensions) == len(target_dimensions)
    return sorted(d % len(shape) for d in target_dimensions)


def _rearrange_dimensions_shapes(
    shape: Shape, split_dimensions: Sequence[int]
) -> Tuple[Shape, Shape]:
    split_shape = tuple(shape[d] for d in split_dimensions)
    remaining_shape = tuple(shape[d] for d in range(len(shape)) if d not in split_dimensions)
    batch_shape = remaining_shape[:-1]
    return split_shape, batch_shape


def _rearrange_dimensions(x: Array, split_dimensions: Sequence[int]) -> Array:
    """Rearrange array so that we can split by a single dimension.

    Turns an array of shape [d1, ..., dn, features] and a list of dimensions to
    split by into [prod(remaining_dimensions), prod(split_dimensions),
    features]

    Args:
        x: array of shape [d1, ..., dn, features]
        split_dimensions: list of dimensions that should end up in dimension -2.

    Returns:
        Rearranged array as described above.
    """
    split_dimensions = [d % len(x.shape) for d in split_dimensions]
    split_dimensions = sorted(split_dimensions)
    split_shape, batch_shape = _rearrange_dimensions_shapes(x.shape, split_dimensions)

    target_dimensions = _target_dimensions(x.shape, split_dimensions)
    x = torch.moveaxis(x, split_dimensions, target_dimensions)

    assert len(x.shape) > len(split_dimensions)
    assert all(isinstance(d, int) and d >= 0 for d in batch_shape)
    assert all(isinstance(d, int) and d >= 0 for d in split_shape)

    if split_shape:
        new_shape = [
            # The use of numpy is okay here, since shapes are concrete at jit time.
            np.prod(batch_shape),
            np.prod(split_shape),
            x.shape[-1],  # features dimension
        ]
    else:
        new_shape = [np.prod(batch_shape), x.shape[-1]]  # features dimension

    res = x.reshape(new_shape)
    return res


def _restore_dimensions(x: Array, original_shape: Shape, split_dimensions: Sequence[int]) -> Array:
    """Restores arrays encoded with _rearrange_dimensions.

    Args:
      x: Array of shape [prod(batch_shape), prod(split_shape), feature...]
      original_shape: Shape of the array to restore to.
      split_dimensions: Dimensions that were multiplied into dimension 2.

    Returns:
      Array of the original shape and axis order for all dimensions in batch_shape
      and split_shape. Feature dimensions may have changed (can include additional
      dimensions for neighbors, for example).
    """
    split_dimensions = [d % len(original_shape) for d in split_dimensions]
    split_dimensions = sorted(split_dimensions)
    split_shape, batch_shape = _rearrange_dimensions_shapes(original_shape, split_dimensions)

    features_shape = x.shape[-2:]  # (num_neighbors, features)
    x = x.reshape((*batch_shape, *split_shape, *features_shape))

    # rearrange
    target_dimensions = _target_dimensions(original_shape, split_dimensions)
    x = torch.moveaxis(x, target_dimensions, split_dimensions)
    return x


class BatchedCache(nn.Module):
    """Equips a cache module with a batch dimension for PyTorch.

    `split_dimensions` indicates the dimensions of the query and update tensors
    that will go to separate caches. By default, we use a separate cache
    for each head. Note that some implementations of the cache share cache
    across all hosts and devices (cache_on_borg, unless configured otherwise)
    or just across devices of each host (cache_on_host). Default is (0,) to
    split by batch only; use (0, -2) to also slit by head dimensions.
    """

    def __init__(self, wrapped: Cache, split_dimensions: Tuple[int] = (0,)):
        super().__init__()
        self.wrapped = wrapped
        self.split_dimensions = split_dimensions

    def update(self, key: Array, value: Array, mask=None):
        """Adds key/value pairs to cache.

        Args:
            key: typically of shape (batch, kv_len, num_heads, k_features). This
                tensor is split up into caches according to `split_dimensions`.
            value: typically of shape (batch, kv_len, num_heads, v_features). This
                tensor is split up into caches according to `split_dimensions`.
                `num_heads` is omitted if attention style is not multi-head.

        Returns:
            A dummy value 0, once the operation has completed.
        """
        key = _rearrange_dimensions(key, self.split_dimensions)

        if value is not None:
            value = _rearrange_dimensions(value, self.split_dimensions)

        if mask is not None:
            mask = _rearrange_dimensions(mask, self.split_dimensions)

        return self.wrapped.update(key, value, mask)

    def topk_retrieval(self, query: Array, mask: Array, num_neighbors: int):
        """Retrieves the nearest neighbors for each query.

        Args:
            query: typically of shape (batch, q_len, num_heads, k_features). This
                tensor is split up into caches according to `split_dimensions`.
            num_neighbors: number of neighbors to retrieve

        Returns:
            Tuple of tensors with the retrieved keys and value of the same shape as
            query, but with an extra dimension of length num_neighbors - typically:
            (batch, q_len, num_heads, num_neighbors, k_features)
        """
        original_shape = query.shape
        if query.ndim == 3:
            query = query.unsqueeze(-2)

        if query.ndim != 4:
            raise ValueError(f"Expected batched inputs; got shape: {query.shape}.")

        query = _rearrange_dimensions(query, self.split_dimensions)

        if mask is not None:
            mask = mask.unsqueeze(-1).repeat(1, 1, query.shape[2])
            mask = _rearrange_dimensions(mask, self.split_dimensions)

        key, value = self.wrapped.topk_retrieval(query, mask, num_neighbors)

        key = _restore_dimensions(key, original_shape, self.split_dimensions)
        if value is not None:
            value = _restore_dimensions(value, original_shape, self.split_dimensions)

        assert key.ndim == len(original_shape) + 1
        return key, value

    def reset(self, caches):
        """Resets the cache.

        Args:
            caches: of shape (num_caches,)

        Returns:
            A dummy value 0, once the operation has completed.
        """
        return self.wrapped.reset(caches)
