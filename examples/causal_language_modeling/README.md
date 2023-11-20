
```
Pretrain a neurocache model on a causal language modeling task


```

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
`LongPile: LP -> 500 Steps`
`LongeRPile: LRP -> 1000 Steps`

---

| Model                 | LP     | LRP    |
| ----------------------| -------| ------ |
| opt-1.3b              | 16.872 | 19.446 | 
| opt-1.3b - lora       | 16.575 | 19.114 |
| opt-1.3b - 16k        | 14.726 | 17.608 |
| opt-1.3b - 128k       | 14.674 | 17.360 |

---

| Model                 | LP    | LRP   |
| ----------------------| ----- | ----- |
| llama2-7b             | 8.102 | 9.075 |
| llama2-7b - lora      | 7.986 | 8.956 |
| llama2-7b - 16k       | 7.454 | 8.589 |
| llama2-7b - 128k      | 7.424 | 8.517 |

---

| Model                 | LP    | LRP   |
| ----------------------| ----- | ----- |
| Mistralv0.1-7b        | 8.308 | 9.432 |
