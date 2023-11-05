import torch

from neurocache.cache import BatchedCache, OnDeviceCache


batch_size = 4
seq_len = 256
num_dims = 128
k = 8
dtype = torch.float16

cache = OnDeviceCache(batch_size, 1024, num_dims, num_dims * 2, 2, dtype=dtype)
cache = BatchedCache(cache)
cache.to("cuda:0")

for segment in range(10):
    query = torch.randn(batch_size, seq_len, num_dims, dtype=dtype, device="cuda:0")
    key = torch.randn(batch_size, seq_len, num_dims, dtype=dtype, device="cuda:0")
    value = torch.randn(batch_size, seq_len, num_dims * 2, dtype=dtype, device="cuda:0")

    start_of_sequence = torch.randint(0, 8, (batch_size,)) == 0
    print("start_of_sequence:", start_of_sequence)
    cache.reset(start_of_sequence)

    topk_keys, top_values = cache.topk_retrieval(query, None, k // 2)
    print("topk_keys.shape:", topk_keys.shape, topk_keys[:, 0, 0, 0])
    print("top_values.shape:", top_values.shape)
    cache.update(key, None, None)

    assert (topk_keys[start_of_sequence, 0, 0] == 0).all(), "Expected all zeros"
