import argparse

from transformers import SchedulerType


def parse_args():
    parser = argparse.ArgumentParser()

    # Task arguments
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--tokenizer_name", type=str)
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--eval_split", type=str, default="validation")
    parser.add_argument("--test_split", type=str, default="test")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
    parser.add_argument("--sequence_length", type=int, default=1024)
    parser.add_argument("--max_train_steps", type=int, default=10000)
    parser.add_argument("--max_eval_steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1234567)

    # Training arguments
    parser.add_argument("--nf4", action="store_true")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--only_evaluate", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--disable_grad_checkpointing", action="store_true")
    # This needs to be > 1 for stable training, otherwuse the model will diverge
    # due to distibutional shift between the cache and the model.
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--evaluate_every_steps", type=int, default=500)
    parser.add_argument("--num_warmup_steps", type=int, default=1000)
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="cosine",
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ],
    )

    # Output arguments
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--resume_from_checkpoint", type=str, default="")
    parser.add_argument("--checkpointing_steps", type=int, default=500)
    parser.add_argument("--keep_n_checkpoints", type=int, default=2)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--report_to", type=str, default="tensorboard")
    parser.add_argument("--with_tracking", type=bool, default=True)

    # Neurocache arguments
    parser.add_argument("--pretrained_neurocache", type=str, default=None)
    parser.add_argument("--disable_neurocache", action="store_true")
    parser.add_argument("--attention_layers", type=str, default=None)
    parser.add_argument("--cache_layers", type=str, default=None)
    parser.add_argument("--cache_size", type=int, default=16384)
    parser.add_argument("--cache_type", type=str, default="FIFO", choices=["FIFO", "LRU"])
    parser.add_argument(
        "--cache_dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"]
    )
    parser.add_argument("--compression_factor", type=int, default=4)
    parser.add_argument("--context_size", type=int, default=2)
    parser.add_argument("--neighborhood_size", type=int, default=2)
    parser.add_argument("--topk", type=int, default=16)

    # LoRA arguments
    parser.add_argument("--disable_lora", action="store_true")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--lora_modules", type=str, default="gate_proj,up_proj,down_proj")

    args = parser.parse_args()
    return args
