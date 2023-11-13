import torch
from tqdm import trange
from transformers import AutoModelForCausalLM, AutoTokenizer

from neurocache import (
    NeurocacheModelForCausalLM,
    OnDeviceCacheConfig,
)


model_name = "facebook/opt-350m"

model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

cache_layer_idx = model.config.num_hidden_layers - 2

config = OnDeviceCacheConfig(
    cache_layers=[
        cache_layer_idx,
    ],
    attention_layers=list(range(cache_layer_idx, model.config.num_hidden_layers)),
    compression_factor=1,
    topk=8,
    cache_size=2048,
)

model = NeurocacheModelForCausalLM(model, config)
model = model.to("cuda")

input_text = "Hello, my dog is cute"
tokenized_input = tokenizer(input_text, return_tensors="pt")
tokenized_input = {k: v.to("cuda") for k, v in tokenized_input.items()}

outputs = model(**tokenized_input)
outputs = model(**tokenized_input)

input_text = [
    """This is a test for the neurocache model. The model should be able to cache the hidden states of the model and use them for the next forward pass. This should speed up the model and reduce the memory consumption."""
    * 10,
    """This is a test for the neurocache model. The model should be able to cache the hidden states of the model and use them for the next forward pass. This should speed up the model and reduce the memory consumption."""
    * 8,
]

tokenized_input = tokenizer(input_text, return_tensors="pt", padding=True)

tokenized_input["start_of_sequence"] = torch.tensor([0, 1]).bool()
tokenized_input = {k: v.to("cuda") for k, v in tokenized_input.items()}

model.reinitialize_cache()

model.eval()
with torch.no_grad():
    for _ in trange(1000):
        outputs = model(**tokenized_input)
        # time.sleep(0.2)

# with model.disable_neurocache():
#     ids = model.generate(**tokenized_input, do_sample=True, num_return_sequences=1)
#     print(tokenizer.batch_decode(ids, skip_special_tokens=True))

# ids = model.generate(**tokenized_input, do_sample=True, num_return_sequences=1)
# print(tokenizer.batch_decode(ids, skip_special_tokens=True))

# model.save_pretrained("bin/test-model")

# model = NeurocacheModelForCausalLM.from_pretrained(
#     AutoModelForCausalLM.from_pretrained(model_name), "bin/test-model"
# )
# model.print_trainable_parameters()

# ids = model.generate(**tokenized_input, do_sample=True, num_return_sequences=1)
# print(tokenizer.batch_decode(ids, skip_special_tokens=True))
