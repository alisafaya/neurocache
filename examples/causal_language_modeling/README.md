# Neurocache adaptation of LLMs

Environment Requirements:
- Python 3.8, 3.9, or 3.10
- torch, transformers, flash_attention, accelerate

Set up the environment:

```
git clone git@github.com:alisafaya/neurocache.git
cd neurocache
pip install -e .
cd examples/causal_language_modeling
pip install -r req.txt
```

## Perplexity results on PG-19 test set

| checkpoint   | OPT-1.3B | Phi-1.5B |
|--------------|----------|----------|
| Original LM  | 25.7880  | 21.9247  |
| +LoRA        | 12.1307  | 17.2815  |
| +cache(16K)  | 11.3762  | 15.6733  |
| +cache(128K) | 11.2845  | 15.5222  |


### Original OPT-1.3B

```bash
TFDS_DATA_DIR=gs://ai-codeway-workspace-shared/shared/nlp/neurostore/tf_datasets \
accelerate launch --num_processes 2 --mixed_precision bf16 train_neurocache.py \
--output_dir ./pg19_opt-1_3b_neurocache/ \
--gradient_acc 4 \
--model_name_or_path facebook/opt-1.3b \
--dataset_name pg19:0.1.1 \
--per_device_train_batch_size 8 \
--per_device_eval_batch_size 8 \
--lora_modules fc1,fc2 \
--max_eval_steps 200 \
--max_train_steps 10000 \
--logging_steps 20 \
--attention_layers '18,19,20,21,22,23' \
--retrieval_map '{18:18}' \
--disable_grad \
--disable_neurocache \
--only_evaluate # Evaluate the original model with zero finetuning/adaptation.

# perplexity: 25.7880, loss: 3.2499, bits_per_token: 4.6886
```

### Adapting OPT-1.3B with LoRA (Neurocache-off)

Training LoRA weights on the PG-19 dataset. We apply LoRA on the FFNs for the models. This will take around 2 hours on a single H100 GPU.

This experiment's results will act as a no-cache baseline to compare against Neurocache adapted OPT-1.3B.

```bash
TFDS_DATA_DIR=gs://ai-codeway-workspace-shared/shared/nlp/neurostore/tf_datasets \
accelerate launch --num_processes 2 --mixed_precision bf16 train_neurocache.py \
--output_dir ./pg19_opt-1_3b_neurocache/ \
--gradient_acc 4 \
--model_name_or_path facebook/opt-1.3b \
--dataset_name pg19:0.1.1 \
--per_device_train_batch_size 8 \
--per_device_eval_batch_size 8 \
--lora_modules fc1,fc2 \
--max_eval_steps 200 \
--max_train_steps 10000 \
--logging_steps 20 \
--attention_layers '18,19,20,21,22,23' \
--retrieval_map '{18:18}' \
--disable_grad \
--disable_neurocache # neurocache is disabled, we only finetune lora's on the same dataset to get fair comparison

perplexity: 12.1307, loss: 2.4957, bits_per_token: 3.6006
``` 

### Adapting OPT-1.3B with Neurocache

```bash
TFDS_DATA_DIR=gs://ai-codeway-workspace-shared/shared/nlp/neurostore/tf_datasets \
accelerate launch --num_processes 2 --mixed_precision bf16 train_neurocache.py \
--output_dir ./pg19_opt-1_3b_neurocache/ \
--gradient_acc 4 \
--model_name_or_path facebook/opt-1.3b \
--dataset_name pg19:0.1.1 \
--per_device_train_batch_size 8 \
--per_device_eval_batch_size 8 \
--lora_modules fc1,fc2 \
--max_eval_steps 200 \
--max_train_steps 10000 \
--logging_steps 20 \
--attention_layers '18,19,20,21,22,23' \
--retrieval_map '{18:18}' \
--disable_grad

# Trained with Cache of 16K
# Evaluation results with 16K tokens cache: perplexity: 11.3762, loss: 2.4315, bits_per_token: 3.5079
# Evaluaton results with 128K tokens cache: perplexity: 11.2845, loss: 2.4234, bits_per_token: 3.4963
```



