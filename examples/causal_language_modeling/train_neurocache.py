"""
Fine-tuning the library models for causal language modeling (llama, opt, ...)
on a text file or a dataset without using HuggingFace Trainer.
"""

import logging
import math
import os

import tensorflow.compat.v2 as tf
import torch
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import recursively_apply, set_seed
from config_args import parse_args
from data_utils import utils as dutils
from peft import (
    LoraConfig,
    inject_adapter_in_model,
)
from torch.nn import functional as F
from tqdm.auto import tqdm
from transformers import (
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    get_scheduler,
)

from neurocache import NeurocacheModelForCausalLM, OnDeviceCacheConfig


# Make sure that tensorflow is not reserving GPUs.
tf.config.experimental.set_visible_devices([], "GPU")

logger = get_logger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())


def initialize_model(args, accelerator):
    config = AutoConfig.from_pretrained(
        args.model_name_or_path,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        config=config,
        low_cpu_mem_usage=args.low_cpu_mem_usage,
    )

    # Apply LoRA only to the main model.
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["fc1", "fc2"],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = inject_adapter_in_model(lora_config, model, "neurocache")

    # Set up the neurocache config and wrap the model with the neurocache model.
    if args.cache_layers is not None:
        args.cache_layers = [int(x) for x in args.cache_layers.split(",")]
    if args.attention_layers is not None:
        args.attention_layers = [int(x) for x in args.attention_layers.split(",")]

    neurocache_config = OnDeviceCacheConfig(
        attention_layers=args.attention_layers,
        cache_layers=args.cache_layers,
        cache_size=args.cache_size,
        cache_dtype=args.cache_dtype,
        compression_factor=args.compression_factor,
        neighborhood_size=args.neighborhood_size,
        context_size=args.context_size,
        topk=args.topk,
    )
    model = NeurocacheModelForCausalLM(model, neurocache_config)

    accelerator.print(lora_config.to_dict())
    accelerator.print(neurocache_config.to_dict())

    print_trainable_parameters(model, accelerator)

    return model


def calculate_metrics(eval_loss):
    try:
        perplexity = math.exp(eval_loss)
    except OverflowError:
        perplexity = float("inf")
    bits_per_token = eval_loss * 1.442695
    return bits_per_token, perplexity


def print_trainable_parameters(model, accelerator):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            accelerator.print(_, param.shape)
            trainable_params += param.numel()
    accelerator.print(
        f"trainable params: {trainable_params} || all params: {all_param} "
        f"|| trainable%: {100 * trainable_params / all_param}"
    )


def initialize_dataloader(args, task_config, split, accelerator):
    distributed_batch_size = args.per_device_batch_size * accelerator.num_processes
    per_device_batch_size = args.per_device_batch_size

    dataset = dutils.LongTextDataset(task_config, split, distributed_batch_size)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, pin_memory=True)

    def slice_fn(data, tensor_slice, process_index=None, num_processes=None):
        """split batch across processes."""
        nonlocal per_device_batch_size

        def _slice_tensor(tensor, tensor_slice):
            # remove the additional batch dim that is added by the dataloader
            return tensor.squeeze(0)[tensor_slice]

        tensor_slice = slice(
            process_index * per_device_batch_size, (process_index + 1) * per_device_batch_size
        )
        return recursively_apply(_slice_tensor, data, tensor_slice)

    # Prepare dataloaders
    dataloader = accelerator.prepare_data_loader(dataloader, slice_fn_for_dispatch=slice_fn)
    return dataloader


def run_evaluation(args, model, dataloader, accelerator, global_step=0, prefix="eval"):
    logger.info("***** Running evaluation *****")
    logger.info(f"    Steps: {global_step}")
    logger.info(f"    Prefix: {prefix}")

    model.eval()
    losses = []
    for step, batch in tqdm(
        enumerate(dataloader),
        desc="Evaluating",
        total=args.max_eval_steps if args.max_eval_steps > 0 else None,
    ):
        with torch.no_grad():
            target_ids = batch.pop("labels")
            epoch = batch.pop("epochs")[0].item()
            outputs = model(**batch)
            loss = F.cross_entropy(
                outputs.logits.view(-1, outputs.logits.size(-1)).to(torch.float32),
                target_ids.view(-1),
                ignore_index=0,  # TODO: get this id from tokenizer
            )

        losses.append(accelerator.gather_for_metrics(loss.repeat(args.per_device_batch_size)))
        if step >= args.max_eval_steps and args.max_eval_steps > 0:
            break

        if epoch > 0:
            logger.warning("Epoch > 0 for evaluation, this is unexpected.")

    eval_loss = torch.cat(losses).mean().item()
    bits_per_token, perplexity = calculate_metrics(eval_loss)

    logger.info(
        f"global_step: {global_step}, perplexity: {perplexity}, loss: {eval_loss}, bits_per_token: {bits_per_token}"
    )

    if args.with_tracking:
        accelerator.log(
            {
                f"{prefix}_perplexity": perplexity,
                f"{prefix}_bits_per_token": bits_per_token,
                f"{prefix}_loss": eval_loss,
            },
            step=global_step,
        )

    return eval_loss, bits_per_token, perplexity


def main():
    args = parse_args()

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator_log_kwargs = {}

    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["project_dir"] = args.output_dir

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        dispatch_batches=True,
        **accelerator_log_kwargs,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # Task configuration
    task_config = dutils.TrainingTaskConfig.from_args(args)

    # Set up seqio tasks
    dutils.setup_tasks(args.model_name_or_path)

    # Initialize the model
    model = initialize_model(args, accelerator)

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # Initialize the datasets
    train_dataloader = initialize_dataloader(
        args, task_config, task_config.train_split, accelerator
    )
    eval_dataloader = initialize_dataloader(args, task_config, task_config.eval_split, accelerator)
    test_dataloader = initialize_dataloader(args, task_config, task_config.test_split, accelerator)

    # Prepare everything with our `accelerator`.
    model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers("clm_no_trainer", experiment_config)

    total_batch_size = (
        args.per_device_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            checkpoint_path = args.resume_from_checkpoint
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
            checkpoint_path = path
            path = os.path.basename(checkpoint_path)

        accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")
        accelerator.load_state(path)

        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        # need to multiply `gradient_accumulation_steps` to reflect real steps
        resume_step = (
            int(training_difference.replace("step_", "")) * args.gradient_accumulation_steps
        )
        starting_epoch = resume_step // len(train_dataloader)
        resume_step -= starting_epoch * len(train_dataloader)
        completed_steps = resume_step // args.gradient_accumulation_steps

    # update the progress_bar if load from checkpoint
    if not args.only_evaluate:
        progress_bar.update(completed_steps)

        model.train()
        if args.resume_from_checkpoint == starting_epoch and resume_step is not None:
            # We skip the first `n` batches in the dataloader when resuming from a checkpoint
            active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
        else:
            active_dataloader = train_dataloader

        if args.with_tracking:
            logging_loss = 0
            logging_steps = 0

        for step, batch in enumerate(active_dataloader):
            with accelerator.accumulate(model):
                target_ids = batch.pop("labels")
                epoch = batch.pop("epochs")[0].item()

                # Forward pass
                outputs = model(**batch)
                loss = F.cross_entropy(
                    outputs.logits.view(-1, outputs.logits.size(-1)).to(torch.float32),
                    target_ids.view(-1),
                    ignore_index=0,  # TODO: get this id from tokenizer
                )

                # Backward pass
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                # Logging
                if args.with_tracking:
                    logging_loss += loss.detach().float()
                    logging_steps += 1

            # Checks if the accelerator has performed
            # an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1

                if args.with_tracking and completed_steps % args.logging_steps == 0:
                    bits_per_token, perplexity = calculate_metrics(logging_loss / logging_steps)
                    accelerator.log(
                        {
                            "train_perplexity": perplexity,
                            "train_bits_per_token": bits_per_token,
                            "train_loss": logging_loss / logging_steps,
                            "train_epoch": epoch,
                            "train_lr": optimizer.param_groups[0]["lr"],
                        },
                        step=completed_steps,
                    )
                    logging_steps, logging_loss = 0, 0
                    logger.info(f"global_step: {completed_steps}, perplexity: {perplexity}")

                if completed_steps % args.evaluate_every_steps == 0:
                    run_evaluation(
                        args,
                        model,
                        eval_dataloader,
                        accelerator,
                        global_step=completed_steps,
                        prefix="eval",
                    )

                # Save model checkpoint
                if completed_steps % args.checkpointing_steps == 0:
                    output_dir = f"step_{completed_steps}"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir)

                if completed_steps >= args.max_train_steps:
                    break

    # Final evaluation
    run_evaluation(args, model, eval_dataloader, accelerator, completed_steps, prefix="eval")
    run_evaluation(args, model, test_dataloader, accelerator, completed_steps, prefix="test")

    if args.with_tracking:
        accelerator.end_training()

    # Save the model
    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            args.output_dir,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
        )


if __name__ == "__main__":
    main()
