Train neurocache weights for huggingface models.

Example:

```bash
accelerate launch --mixed_precision fp16 \
    train_neurocache.py --output_dir /localscratch/proj12/nc/baseline_opt350m/ \
    --model_name_or_path facebook/opt-350m \
    --dataset_name long_pile:1.1.0 \
    --sequence_length 1024 \
    --max_eval_steps 1000 \
    --evaluate_every_steps 1000 \
    --checkpointing_steps 5000 \
    --with_tracking \
    --gradient_accumulation_steps 4 \
    --per_device_batch_size 8 &> /localscratch/proj12/nc/baseline_opt350m/log.txt &
``` 

## Tuning OPT-1.3B

```sh
accelerate launch \
  --num_processes 4 \
  --mixed_precision bf16 \
  train_neurocache.py \
  --output_dir /localscratch/proj12/nc/lp1.1_opt1.3b_final/ \
  --model_name_or_path facebook/opt-1.3b \
  --dataset_name long_pile:1.1.0 \
  --disable_grad_checkpointing \
  --gradient_accumulation_steps 2 \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 4 \
  --lora_modules fc1,fc2 \
  --cache_size 16384 \
  --cache_dtype bfloat16 \
  --with_tracking \
  --nf4
```

### Results

`bsz=32, seq_len=1024`
`LongLongPile: LP -> 1000 Steps`
`Project Gutenberg-19: PG -> 100 Steps`

---

| Model                 | LP     | LRP    | PG19   |
| ----------------------| -------| ------ | ------ |
| opt-1.3b              | 16.872 | 19.446 | 25.859 |
| opt-1.3b - 16k        | 14.764 | 17.626 | ------ |
| opt-1.3b - 128k       | 14.711 | 17.377 | ------ |
| opt-1.3b - lora       | 16.575 | 19.114 | 12.199 |
| opt-1.3b - 16k+lora   | 14.726 | 17.608 | 11.306 |
| opt-1.3b - 128k+lora  | 14.674 | 17.360 | 11.227 |
 
---

`seq_len=2048`

| Model                 | LP     |
| ----------------------| -------|
| opt-1.3b              | 15.649 |
| opt-1.3b - 16k - 512  | 14.263 | 
| opt-1.3b - 16k - 1024 | 14.240 | 
| opt-1.3b - 16k - 2048 | 14.617 |

`seq_len=4096`

| Model                 | LRP   |
| ----------------------| ------|
| llama2-7b             | 8.140 |
| llama2-7b - 16k       | 7.915 |
| llama2-7b - 128k      | 7.853 |

---

| Model                   | LP     | LRP   | PG19  |
| ----------------------- | ------ | ----- | ----- |
| llama2-7b               | 8.102  | 9.075 | 7.359 |
| llama2-7b - lora        | 7.986  | 8.956 | 7.073 |
| llama2-7b - 16k         | 7.393  | 8.451 | ----- |
| llama2-7b - 128k        | 7.360  | 8.363 | ----- |
| llama2-7b - 16k+lora    | 7.345  | 8.435 | 7.117 |
| llama2-7b - 128k+lora   | 7.312  | 8.347 | 7.078 |
| llama2-7b - 2*16k+lora  | -      | 8.307 | 7.003 |
| llama2-7b - 2*128k+lora | -      | 8.219 | 6.959*|

(*) Evaluate using model trained on LP instead of PG19.

| Model                   |  LRP   | PG19  |
| ----------------------- | ------ | ----- |
| mistral-7b              | 9.380  | 7.863 |
| mistral-7b - 16k+lora   | 8.581  | 7.684*|
| mistral-7b - 128k+lora  | 8.493  | 7.636*|

---

Cache ordering: llama2-7b - 16k

| Order                     | LRP   |
| ------------------------- | ----- |
| FIFO/training - FIFO/test | 8.451 |
| FIFO/training - LRU/test  | 8.508 |



| default   | 8.581 |
| topk = 32 | 8.573 |
| w = 4     | 8.592 |
| c = 4     | |
