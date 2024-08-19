# Neurocache adaptation of LLMs

## Evaluating Phi-1.5 on PG-19 Test set

| Phi-1.5     | Perplexity |
|-------------|------------|
| Original LM | 20.2914    |
| +LoRA       | | 
| +cache(16K) | |

### Original Phi-1.5

```bash
TFDS_DATA_DIR=gs://ai-codeway-workspace-shared/shared/nlp/neurostore/tf_datasets accelerate launch \
    --mixed_precision bf16 train_neurocache.py \
    --output_dir ./pg19_phi-1_5_neurocache/ \
    --model_name_or_path microsoft/phi-1_5 \
    --dataset_name pg19:0.1.1 \
    --per_device_eval_batch_size 8 \
    --max_eval_steps 500 \
    --sequence_length 2048 \
    --logging_steps 50 \
    --disable_neurocache \
    --disable_lora \
    --only_evaluate

perplexity: 20.2914, loss: 3.0101, bits_per_token: 4.34279
```

### Finetuning Phi-1.5 with LoRA

Training LoRA weights on the PG-19 dataset. We apply LoRA on the FFNs for the models. This will take around 2 hours on a single H100 GPU.

```bash
TFDS_DATA_DIR=gs://ai-codeway-workspace-shared/shared/nlp/neurostore/tf_datasets accelerate launch \
    --mixed_precision bf16 train_neurocache.py \
    --output_dir ./pg19_microsoft_phi-1_5_neurocache/ \
    --model_name_or_path microsoft/phi-1_5 \
    --dataset_name pg19:0.1.1 \
    --gradient_accumulation_steps 2 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 8 \
    --lora_modules fc1,fc2 \
    --max_eval_steps 500 \
    --max_train_steps 1000 \
    --sequence_length 2048 \
    --num_warmup_steps 200 \
    --logging_steps 50 \
    --disable_neurocache

``` 

