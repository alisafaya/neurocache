# ## Fine-tune large models using 🤗 `peft` adapters, `transformers` & `bitsandbytes`
#
# In this tutorial we will cover how we can fine-tune large language models using the very recent `peft` library and `bitsandbytes` for loading large models in 8-bit.
# The fine-tuning method will rely on a recent method called "Low Rank Adapters" (LoRA), instead of fine-tuning the entire model you just have to fine-tune these adapters and load them properly inside the model.
# After fine-tuning the model you can also share your adapters on the 🤗 Hub and load them very easily. Let's get started!

# ### Install requirements
#
# First, run the cells below to install the requirements:

# !pip install -q bitsandbytes datasets accelerate
# !pip install -q git+https://github.com/huggingface/transformers.git@main git+https://github.com/huggingface/peft.git

# ### Model loading
#
# Here let's load the `opt-6.7b` model, its weights in half-precision (float16) are about 13GB on the Hub! If we load them in 8-bit we would require around 7GB of memory instead.


from peft import (
    LoraConfig,
    inject_adapter_in_model,
)
from transformers import AutoModelForCausalLM, AutoTokenizer

from neurocache import NeurocacheModelForCausalLM, OnDeviceCacheConfig


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            print(_, param.shape)
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} "
        f"|| trainable%: {100 * trainable_params / all_param}"
    )


model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")  # , load_in_8bit=True)
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")

"""
Prepare model for training:
    Some pre-processing needs to be done before training such an int8 model using `peft`, therefore let's import an utiliy function `prepare_model_for_int8_training` that will:
    - Casts all the non `int8` modules to full precision (`fp32`) for stability
    - Add a `forward_hook` to the input embedding layer to enable gradient computation of the input hidden states
    - Enable gradient checkpointing for more memory-efficient training
"""
# model = prepare_model_for_kbit_training(model)

"""
Apply LoRA only to the main model and train the cache weights fully.

- Memory LoRA: ['fc1', 'fc2']
- Downstream LoRA: ['q_proj', 'v_proj']
"""
config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["fc1", "fc2"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = inject_adapter_in_model(config, model, adapter_name="neurocache")

"""
Set up the neurocache config and wrap the model with the neurocache model.
"""
neurocache_config = OnDeviceCacheConfig(
    cache_layers=[9],
    cache_size=2048,
    cache_dtype="bfloat16",
    attention_layers=[9, 10, 11],
    compression_factor=4,
    topk=8,
)
model = NeurocacheModelForCausalLM(model, neurocache_config)
model = model.to("cuda")

"""
Train the model on a small dataset.
"""
print_trainable_parameters(model)

import transformers
from datasets import load_dataset


data = load_dataset("Abirate/english_quotes")
data = data.map(
    lambda samples: tokenizer(
        samples["quote"],
        truncation=True,
        max_length=128,
        pad_to_multiple_of=8,
        padding="max_length",
    ),
    batched=True,
)

# Please re-enable for inference!
# Silence the warnings.
model.config.use_cache = False

trainer = transformers.Trainer(
    model=model,
    train_dataset=data["train"],
    args=transformers.TrainingArguments(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        warmup_steps=100,
        max_steps=100,
        learning_rate=1e-4,
        fp16=True,
        logging_steps=10,
        output_dir="outputs",
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
trainer.train()

"""
Save the adapters and the weights of the cache.
"""
model.save_pretrained("./opt_peft_model")
model = NeurocacheModelForCausalLM.from_pretrained(model, "./opt_peft_model")
