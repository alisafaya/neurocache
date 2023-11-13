
```
Pretrain a neurocache model on a causal language modeling task

options:
  -h, --help            show this help message and exit
  --dataset_name DATASET_NAME
                        The name of the dataset to use (via the tensorflow datasets).
  --model_name_or_path MODEL_NAME_OR_PATH
                        Path to pretrained model or model identifier from huggingface.co/models.
  --tokenizer_path TOKENIZER_PATH
                        Pretrained tokenizer name or path if not the same as model_name
  --per_device_train_batch_size PER_DEVICE_TRAIN_BATCH_SIZE
                        Batch size (per device) for the training dataloader.
  --per_device_eval_batch_size PER_DEVICE_EVAL_BATCH_SIZE
                        Batch size (per device) for the evaluation dataloader.
  --learning_rate LEARNING_RATE
                        Initial learning rate (after the potential warmup period) to use.
  --weight_decay WEIGHT_DECAY
                        Weight decay to use.
  --max_train_steps MAX_TRAIN_STEPS
                        Total number of training steps to perform. If provided, overrides num_train_epochs.
  --gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS
                        Number of updates steps to accumulate before performing a backward/update pass.
  --lr_scheduler_type {linear,cosine,cosine_with_restarts,polynomial,constant,constant_with_warmup}
                        The scheduler type to use.
  --num_warmup_steps NUM_WARMUP_STEPS
                        Number of steps for the warmup in the lr scheduler.
  --output_dir OUTPUT_DIR
                        Where to store the final model.
  --seed SEED           A seed for reproducible training.
  --sequence_length SEQUENCE_LENGTH
                        Optional input sequence length after tokenization. The training dataset will be truncated in block of this size for training. Default to the model max input length for single sentence inputs (take into account
                        special tokens).
  --window_size WINDOW_SIZE
                        Optional window size. This is used for models with sliding window attention layers.
  --checkpointing_steps CHECKPOINTING_STEPS
                        Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.
  --resume_from_checkpoint RESUME_FROM_CHECKPOINT
                        If the training should continue from a checkpoint folder.
  --with_tracking       Whether to enable experiment trackers for logging.
  --report_to REPORT_TO
                        The integration to report the results and logs to. Supported platforms are `"tensorboard"`, `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations. Only applicable when `--with_tracking` is passed.
  --low_cpu_mem_usage   It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded.If passed, LLM loading time and RAM consumption will be benefited.
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
