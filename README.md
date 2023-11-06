
<h1 align="center">Neurocache</h1>
<h3 align="center">
  A library for augmenting language models with external caching mechanisms 
</h3>

<a href="https://github.com/alisafaya/neurocache/releases">
  <img alt="GitHub release" src="https://img.shields.io/github/release/alisafaya/neurocache.svg">
</a>

## Requirements

* Python 3.6+
* PyTorch 1.13.0+
* Transformers 4.25.0+

## Installation

```bash
pip install neurocache
```

## Getting started

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from neurocache import (
    NeurocacheModelForCausalLM,
    OnDeviceCacheConfig,
)

model_name = "facebook/opt-350m"

model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

cache_layer_idx = model.config.num_hidden_layers - 5

config = OnDeviceCacheConfig(
    cache_layers=[cache_layer_idx, cache_layer_idx + 3],
    attention_layers=list(range(cache_layer_idx, model.config.num_hidden_layers)),
    compression_factor=8,
    topk=8,
)

model = NeurocacheModelForCausalLM(model, config)

input_text = ["Hello, my dog is cute", "Hello, my cat is cute"]
tokenized_input = tokenizer(input_text, return_tensors="pt")
tokenized_input["start_of_sequence"] = torch.tensor([0, 1]).bool()

outputs = model(**tokenized_input)
```

## Supported model types

```
from neurocache.utils import NEUROCACHE_SUPPORTED_MODELS
print(NEUROCACHE_SUPPORTED_MODELS)

[
  "opt",
  "llama",
  "mistral",
  "gptj",
]
```