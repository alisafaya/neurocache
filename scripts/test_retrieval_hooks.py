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

input_text = "Hello, my dog is cute"
tokenized_input = tokenizer(input_text, return_tensors="pt")

outputs = model(**tokenized_input)
outputs = model(**tokenized_input)

input_text = ["Hello, my dog is cute", "Hello, my cute cat"]
tokenized_input = tokenizer(input_text, return_tensors="pt", padding=True)

model.reinitialize_cache()

outputs = model(**tokenized_input)
outputs = model(**tokenized_input)

tokenized_input["start_of_sequence"] = torch.tensor([0, 1]).bool()
outputs = model(**tokenized_input)

with model.disable_neurocache():
    ids = model.generate(**tokenized_input, do_sample=True, num_return_sequences=1)
    print(tokenizer.batch_decode(ids, skip_special_tokens=True))

ids = model.generate(**tokenized_input, do_sample=True, num_return_sequences=1)
print(tokenizer.batch_decode(ids, skip_special_tokens=True))

model.save_pretrained("bin/test-model")

model = NeurocacheModelForCausalLM.from_pretrained(
    AutoModelForCausalLM.from_pretrained(model_name), "bin/test-model"
)
model.print_trainable_parameters()

ids = model.generate(**tokenized_input, do_sample=True, num_return_sequences=1)
print(tokenizer.batch_decode(ids, skip_special_tokens=True))
