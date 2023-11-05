import functools
from typing import Callable, Sequence, Tuple, TypeVar, Union

import numpy as np
import torch

from .cache_utils import Cache


Shape = Sequence[int]
Dtype = torch.dtype
Array = torch.Tensor

Axes = Union[int, Tuple[int, ...]]
F = TypeVar("F", bound=Callable)


def batched_topk(
    query: Array, key: Array, num_buckets: int, bucket_size: int, scoring_fn: str
) -> Tuple[Array, Array]:
    num_queries, _ = query.shape
    num_keys, key_features = key.shape

    # Prevent having an array with prod(dims) > 2147483647
    assert num_queries * num_keys < torch.iinfo(torch.int32).max

    def batch_dot(_query, _batch):
        _scores = torch.einsum("qf,df->qd", _query, _batch)
        _mask = (_batch.sum(-1) == 0).type(torch.bfloat16) * -1e8
        _scores += _mask
        return _scores

    def batch_l2(_query, _batch):
        _batch_squared_sum = -torch.sum(torch.square(_batch), axis=-1)
        _batch_squared_sum = torch.unsqueeze(_batch_squared_sum, axis=0)
        _scores = _batch_squared_sum + 2 * torch.einsum("qf,df->qd", _query, _batch)
        _mask = (_batch.sum(-1) == 0).type(torch.bfloat16) * -1e8
        _scores += _mask
        return _scores

    if scoring_fn == "dot":
        batch_score_fn = functools.partial(batch_dot, query)
    elif scoring_fn == "l2":
        batch_score_fn = functools.partial(batch_l2, query)
    else:
        raise ValueError("Top-k scoring function should be either `dot` or `l2`")

    key = torch.reshape(key, (bucket_size, num_buckets, key_features))
    scores = torch.stack([batch_score_fn(batch) for batch in key], dim=1)

    topk_scores, topk_indices = torch.max(scores, dim=1)
    topk_indices = topk_indices * num_buckets + torch.arange(
        num_buckets, device=key.device
    ).reshape(1, num_buckets)
    return topk_scores, topk_indices


def retrieve_topk_gatherless(
    query: Array,
    key: Array,
    value: Array,
    num_neighbors: int,
    scoring_fn: str = "l2",
    neighborhood_size: int = 1,
) -> Tuple[Array, Array, Array, Array]:
    num_kv, query_features = query.shape
    cache_size, key_features = key.shape
    assert query_features == key_features

    num_buckets = min(max(num_neighbors, 512), cache_size)
    bucket_size = cache_size // num_buckets

    if num_buckets > cache_size:
        raise ValueError(f"More buckets than items in cache. {num_buckets} > {cache_size}")

    if cache_size % num_buckets:
        raise ValueError(f"Buckets must divide cache: {cache_size} % {num_buckets}.")

    # Get topk_indices and topk_scores
    max_bsize = 2 ** np.log2(np.iinfo(np.int32).max // cache_size).astype(int)
    if num_kv < max_bsize:
        topk_scores, topk_indices = batched_topk(query, key, num_buckets, bucket_size, scoring_fn)
    else:
        query = query.reshape(-1, max_bsize, query_features)
        results = [batched_topk(q, key, num_buckets, bucket_size, scoring_fn) for q in query]
        topk_scores, topk_indices = zip(*results)
        topk_indices = torch.reshape(torch.stack(topk_indices), (num_kv, -1))
        topk_scores = torch.reshape(torch.stack(topk_scores), (num_kv, -1))

    if num_buckets > num_neighbors:
        # If we have more buckets than neighbors, we need to do another topk
        # This happens when the cache size is very large: Hierarchical topk.
        topk_scores, topk_buckets = torch.topk(topk_scores, num_neighbors, dim=1)
        topk_indices = torch.gather(topk_indices, 1, topk_buckets)

    if neighborhood_size > 1:
        # Fetch neighboring keys and values of the topk_indices
        # Get the indices of the neighbors
        neighborhood = torch.arange(
            -(neighborhood_size // 2) + 1,
            (neighborhood_size // 2) + 1,
            device=topk_indices.device,
        )
        topk_indices = topk_indices[..., None] + neighborhood[None, :]
        # make sure we don't go out of bounds
        topk_indices = torch.clip(topk_indices, 0, cache_size - 1)

        # re-adjust the shape of the indices
        # from: (num_queries, num_neighbors, neighborhood_size)
        # to: (num_queries, num_neighbors * neighborhood_size)
        topk_indices = topk_indices.reshape(num_kv, -1)

    selected_keys = torch.index_select(key, 0, topk_indices.flatten()).reshape(
        num_kv, -1, key_features
    )

    if value is not None:
        selected_values = torch.index_select(value, 0, topk_indices.flatten()).reshape(
            num_kv, -1, value.shape[-1]
        )
    else:
        selected_values = None

    return selected_keys, selected_values, topk_scores, topk_indices


class OnDeviceCache(Cache):
    def __init__(
        self,
        num_caches: int,
        cache_size: int,
        key_features: int = 128,
        value_features: int = 0,
        neighborhood_size: int = 1,
        scoring_fn: str = "l2",
        dtype: Dtype = torch.float32,
        ordering: str = "FIFO",
    ):
        super(OnDeviceCache, self).__init__(num_caches)

        assert (
            np.prod((num_caches, cache_size, key_features)) < np.iinfo(np.int32).max
        ), "Database size too large for int32 indexing. Reduce batchsize or cache size."

        self.cache_size = cache_size
        self.key_features = key_features
        self.value_features = value_features
        self.neighborhood_size = neighborhood_size
        self.scoring_fn = scoring_fn
        self.dtype = dtype
        self.ordering = ordering

        assert self.neighborhood_size < self.cache_size, (
            f"{self.neighborhood_size} vs {self.cache_size}"
            " Neighborhood size must be smaller than cache size."
        )

        assert (
            self.neighborhood_size % 2 == 0 or self.neighborhood_size == 1
        ), f"{self.neighborhood_size} Neighborhood size must be even."

        self.register_buffer(
            "db_index", torch.zeros(self.num_caches, dtype=torch.int32), persistent=False
        )
        self.register_buffer(
            "key_db",
            torch.zeros(
                self.num_caches,
                self.cache_size,
                self.key_features,
                dtype=self.dtype,
            ),
            persistent=False,
        )

        if self.value_features > 0:
            self.register_buffer(
                "value_db",
                torch.zeros(
                    self.num_caches,
                    self.cache_size,
                    self.value_features,
                    dtype=self.dtype,
                ),
                persistent=False,
            )
        else:
            self.value_db = None

    def update_kv_cache_(self, cache, new_values, start_index):
        num_caches, cache_size, _ = cache.shape
        assert cache_size == self.cache_size, f"{cache_size} vs {self.cache_size}"
        assert num_caches == self.num_caches
        assert new_values.ndim == 3
        assert start_index.shape == (self.num_caches,)

        if self.ordering == "FIFO":
            # FIFO: overwrite oldest entries first
            update_indices = (
                torch.arange(new_values.shape[1], device=start_index.device) + start_index[:, None]
            )  # (num_caches, num_kv)
            cache.scatter_(1, update_indices[..., None].expand_as(new_values), new_values)
        elif self.ordering == "LRU":
            # LRU: overwrite least recently used entries first
            # TODO: implement LRU
            raise NotImplementedError
        else:
            raise ValueError(f"Unknown ordering: {self.ordering}")

    def update(self, key: Array, value: Array = None, mask: Array = None):
        """Add keys and values to the cache; overwrite if cache is full."""
        key = key.detach().to(self.key_db.device)
        if key.ndim == 2:
            assert self.num_caches == 1
            key = key.unsqueeze(1)
        else:
            assert key.ndim == 3

        num_kv, num_caches, key_features = key.shape
        assert num_caches == self.num_caches
        assert key_features == self.key_features

        key = torch.moveaxis(key, source=1, destination=0)  # split by cache

        # start_index can be larger than DB - we use that to detect which entries
        # are not written to yet
        start_index = self.db_index % self.cache_size
        self.update_kv_cache_(self.key_db, key, start_index)

        if value is not None:
            value = value.detach().to(self.value_db.device)
            assert value.shape[-1] == self.value_features
            if value.ndim == 2:
                assert self.num_caches == 1
                value = value.unsqueeze(1)
            else:
                assert value.ndim == 3
            value = torch.moveaxis(value, source=1, destination=0)  # split by cache
            self.update_kv_cache_(self.value_db, value, start_index)

        self.db_index = self.db_index + num_kv
        return 0

    def reset(self, caches: Array) -> int:
        """Resets specified caches."""
        caches = caches.detach()

        assert caches.shape == (self.num_caches,)
        assert caches.dtype == torch.bool

        # Reset key_db, db_index for the specified caches
        self.db_index[caches] = 0
        self.key_db[caches] = 0.0

        # Reset value_db for the specified caches (if it exists)
        if self.value_db is not None:
            self.value_db[caches] = 0.0
        return 0

    def topk_retrieval(self, query: Array, mask: Array, num_neighbors: int) -> Tuple[Array, Array]:
        """Nearest neighbors by full multiplication and approximate top k on GPU."""
        query = query.detach().to(self.key_db.device)
        if query.ndim == 2:
            assert self.num_caches == 1
            query = query.unsqueeze(1)
        else:
            assert query.ndim == 3
            assert query.shape[1] == self.num_caches, (
                f"{query.shape[1]} vs {self.num_caches}"
                " Number of caches in query does not match cache."
            )

        unused_num_kv, unused_num_caches, query_features = query.shape
        assert query_features == self.key_features
        query = torch.movedim(query, source=1, destination=0)

        # Process batches sequentially
        selected_keys, selected_values, _, _ = zip(
            *(
                retrieve_topk_gatherless(
                    query[i],
                    self.key_db[i],
                    self.value_db[i] if self.value_db is not None else None,
                    num_neighbors,
                    self.scoring_fn,
                    self.neighborhood_size,
                )
                for i in range(query.shape[0])
            )
        )

        selected_keys = torch.stack(selected_keys)
        assert selected_keys.ndim == 4
        selected_keys = torch.movedim(selected_keys, source=0, destination=1)

        if selected_values[0] is not None:
            selected_values = torch.stack(selected_values)
            assert selected_values.ndim == 4
            selected_values = torch.movedim(selected_values, source=0, destination=1)
        else:
            assert self.value_db is None
            selected_values = None

        return selected_keys, selected_values
