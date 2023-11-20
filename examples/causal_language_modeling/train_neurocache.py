"""
Fine-tuning the library models for causal language modeling (llama, opt, ...)
on a text file or a dataset without using HuggingFace Trainer.
"""

import logging
import math
import os
import shutil

import tensorflow.compat.v2 as tf
import torch
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import recursively_apply, set_seed
from peft import LoraConfig, inject_adapter_in_model, prepare_model_for_kbit_training
from torch.nn import functional as F
from tqdm.auto import tqdm
from transformers import (
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    get_scheduler,
)
from utils import data_utils as dutils
from utils.args import parse_args

from neurocache import NeurocacheModelForCausalLM, OnDeviceCacheConfig
from neurocache.neurocache import get_attribute


# Make sure that tensorflow is not reserving GPUs.
tf.config.experimental.set_visible_devices([], "GPU")


logger = get_logger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())


def init_neurocache_weights_(args, neurocache):
    _model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        config=AutoConfig.from_pretrained(
            args.model_name_or_path,
        ),
    )

    if hasattr(_model, "base_model"):
        _model = _model.model
    if hasattr(_model, "model"):
        _model = _model.model
    if hasattr(_model, "decoder"):
        _model = _model.decoder
    layers = get_attribute(_model, "layers")

    for i, idx in enumerate(neurocache.config.attention_layers):
        self_attn_module = get_attribute(layers[idx], "self_attn")
        neurocache.cache_attns[i]._init_proj_weights(self_attn_module)

    del _model


def prevent_full_backward_pass(_model):
    """
    Allow gradient checkpointing while preventing full backward pass on the model.
    """

    # Prevent full backward pass on the model. This is a workaround for the issue that
    # the full backward pass on the model will cause the memory and GPU usage to increase.

    # This is required because the default implementation of transformers does checkpointing
    # on all of the layers by default. This cause the backward pass to be done on all of the
    # layers, which is not necessary for our use case.

    # We fix this by enabling gradient checkpointing on the model, and then detaching the
    # output of the bottom layer. This will prevent the backward pass from propagating to
    # the bottom layer, and thus prevent the full backward pass on the model.

    # Note: this does not work by default due to PyTorch checkpoint function default use_reentrant
    # parameter to be True. We modify it to false in the transformers library.
    # modeling_utils.py:
    # After this line:
    # from torch.utils.checkpoint import checkpoint
    # Add this line:
    # checkpoint = functools.partial(checkpoint, use_reentrant=False)

    neurocache_config = _model.base_cache.config
    bottom_layer_idx = min(neurocache_config.attention_layers + neurocache_config.cache_layers)

    if hasattr(_model, "base_model"):
        _model = _model.model
    if hasattr(_model, "model"):
        _model = _model.model
    if hasattr(_model, "decoder"):
        _model = _model.decoder
    layers = get_attribute(_model, "layers")

    def forward_hook(module, args, output):
        return tuple(a.detach() if isinstance(a, torch.Tensor) else a for a in output)

    bottom_layer = layers[bottom_layer_idx - 1]
    bottom_layer.register_forward_hook(forward_hook)


def initialize_model(args, accelerator):
    if args.nf4:
        compute_dtype = getattr(torch, args.cache_dtype)
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            device_map="auto",
        )
    else:
        quant_config = None

    config = AutoConfig.from_pretrained(
        args.model_name_or_path,
    )
    config.use_cache = False
    config._flash_attn_2_enabled = True

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        config=config,
        quantization_config=quant_config,
        torch_dtype=getattr(torch, args.cache_dtype) if not args.nf4 else None,
    )

    # Prepare the model for kbit training.
    if args.nf4:
        model = prepare_model_for_kbit_training(model)

    # Set up the neurocache config and wrap the model with the neurocache model.
    if args.cache_layers is not None:
        cache_layers = [int(x) for x in args.cache_layers.split(",")]
    else:
        cache_layers = [model.config.num_hidden_layers * 3 // 4]

    if args.attention_layers is not None:
        attention_layers = [int(x) for x in args.attention_layers.split(",")]
    else:
        attention_layers = list(range(min(cache_layers), model.config.num_hidden_layers))

    if not args.disable_lora:
        # Apply LoRA to the main model to adapt it to using neurocache.
        lora_layers = attention_layers
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=args.lora_modules.split(","),
            lora_dropout=args.lora_dropout,
            layers_to_transform=lora_layers,
            bias="none",
            task_type="CAUSAL_LM",
        )
        logger.info(f"LoRA Config: {lora_config}")
        model = inject_adapter_in_model(lora_config, model, "neurocache")

    if not args.disable_neurocache:
        if not args.pretrained_neurocache:
            neurocache_config = OnDeviceCacheConfig(
                attention_layers=attention_layers,
                cache_layers=cache_layers,
                cache_size=args.cache_size,
                cache_dtype=args.cache_dtype,
                compression_factor=args.compression_factor,
                neighborhood_size=args.neighborhood_size,
                context_size=args.context_size,
                topk=args.topk,
            )
            model = NeurocacheModelForCausalLM(model, neurocache_config)
            logger.info(f"Neurocache Config: {neurocache_config}")
            logger.info("Initializing neurocache weights from pretrained model.")
            init_neurocache_weights_(args, model.base_cache)
        else:
            logger.info(f"Loading neurocache from {args.pretrained_neurocache}")
            model = NeurocacheModelForCausalLM.from_pretrained(
                model, args.pretrained_neurocache, cache_type="ONDEVICE", is_training=True
            )
        model = model.to(accelerator.device)

    if args.disable_grad_checkpointing:
        model.base_model.gradient_checkpointing_disable()
    else:
        model.base_model.gradient_checkpointing_enable()
        model.base_model.enable_input_require_grads()
        prevent_full_backward_pass(model)

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


def initialize_dataloader(batch_size, task_config, split, accelerator, resume_step=0):
    distributed_batch_size = batch_size * accelerator.num_processes
    per_device_batch_size = batch_size

    dataset = dutils.LongTextDataset(task_config, split, distributed_batch_size, skip=resume_step)
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

    if not args.disable_neurocache:
        if isinstance(model, NeurocacheModelForCausalLM):
            model.base_cache.reinitialize_cache()
        elif isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model.module.base_cache.reinitialize_cache()

    model.eval()
    losses = []
    progress_bar = tqdm(
        range(args.max_eval_steps if args.max_eval_steps > 0 else None),
        disable=not accelerator.is_local_main_process,
    )

    for step, batch in enumerate(dataloader):
        with torch.no_grad():
            target_ids = batch.pop("labels")
            epoch = batch.pop("epochs")[0].item()

            if args.disable_neurocache:
                batch.pop("start_of_sequence")

            outputs = model(**batch)
            loss = F.cross_entropy(
                outputs.logits.view(-1, outputs.logits.size(-1)).to(torch.float32),
                target_ids.view(-1),
                ignore_index=0,  # TODO: get this id from tokenizer
            )

        losses.append(accelerator.gather_for_metrics(loss.repeat(args.per_device_eval_batch_size)))
        if step >= args.max_eval_steps and args.max_eval_steps > 0:
            break

        if epoch > 0:
            logger.warning("Epoch > 0 for evaluation, this is unexpected.")

        progress_bar.update(1)

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

    model.train()
    if not args.disable_neurocache:
        if isinstance(model, NeurocacheModelForCausalLM):
            model.base_cache.reinitialize_cache()
        elif isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model.module.base_cache.reinitialize_cache()
    return eval_loss, bits_per_token, perplexity


def main():
    args = parse_args()

    print(args)

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
        split_batches=True,
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
                p
                for n, p in model.named_parameters()
                if (not any(nd in n for nd in no_decay) and p.requires_grad)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if (any(nd in n for nd in no_decay) and p.requires_grad)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # Initialize the datasets
    eval_dataloader = initialize_dataloader(
        args.per_device_eval_batch_size, task_config, task_config.eval_split, accelerator
    )

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
        args.per_device_train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)

    # Potentially load in the weights and states from a previous save
    completed_steps, resume_step = 0, 0
    if args.resume_from_checkpoint:
        checkpoint_path = args.resume_from_checkpoint.rstrip("/")

        accelerator.load_state(checkpoint_path)
        accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")

        # need to multiply `gradient_accumulation_steps` to reflect real steps
        training_difference = os.path.splitext(os.path.basename(checkpoint_path))[0]
        resume_step = (
            int(training_difference.replace("step_", "")) * args.gradient_accumulation_steps
        )
        completed_steps = resume_step // args.gradient_accumulation_steps

    # update the progress_bar if load from checkpoint
    if not args.only_evaluate:
        train_dataloader = initialize_dataloader(
            args.per_device_train_batch_size,
            task_config,
            task_config.train_split,
            accelerator,
            resume_step=resume_step,
        )

        progress_bar.update(completed_steps)

        model.train()
        logging_loss, logging_steps = 0, 0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                target_ids = batch.pop("labels")
                epoch = batch.pop("epochs")[0].item()

                if args.disable_neurocache:
                    batch.pop("start_of_sequence")

                # Forward pass
                outputs = model(**batch)
                loss = F.cross_entropy(
                    outputs.logits.view(-1, outputs.logits.size(-1)).to(torch.float32),
                    target_ids.view(-1),
                    ignore_index=0,  # TODO: get this id from tokenizer
                )

                # Backward pass
                accelerator.backward(loss)

                # Clip gradients
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                # Logging
                logging_loss += loss.detach().float()
                logging_steps += 1

            # Checks if the accelerator has performed
            # an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1

                if completed_steps % args.logging_steps == 0:
                    bits_per_token, perplexity = calculate_metrics(logging_loss / logging_steps)
                    # Log metrics with formatting floats
                    progress_bar.set_description(
                        f"step {completed_steps:5d}, "
                        f"train_loss = {logging_loss / logging_steps:.4f}, "
                        f"train_perplexity = {perplexity:.4f}, "
                        f"learning_rate = {optimizer.param_groups[0]['lr']:.4e}"
                    )

                    if args.with_tracking:
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

                if completed_steps % args.evaluate_every_steps == 0:
                    run_evaluation(
                        args,
                        model,
                        eval_dataloader,
                        accelerator,
                        global_step=completed_steps,
                        prefix="validation",
                    )

                # Save model checkpoint
                if completed_steps % args.checkpointing_steps == 0:
                    output_dir = f"step_{completed_steps}"
                    output_dir = os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir)

                    # keep only last n checkpoints
                    if args.keep_n_checkpoints is not None:
                        checkpoint_paths = [
                            os.path.join(args.output_dir, path)
                            for path in os.listdir(args.output_dir)
                            if path.startswith("step_")
                        ]
                        checkpoint_paths = sorted(
                            checkpoint_paths, key=lambda path: int(path.split("_")[-1])
                        )
                        for ckpt in checkpoint_paths[: -args.keep_n_checkpoints]:
                            logger.info(f"Removing old checkpoint: {ckpt}")
                            # remove checkpoint
                            shutil.rmtree(ckpt, ignore_errors=True)

                if completed_steps >= args.max_train_steps:
                    break

        if args.with_tracking:
            accelerator.end_training()

        # Save the model
        if args.output_dir is not None:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            if accelerator.is_main_process:
                if isinstance(unwrapped_model, NeurocacheModelForCausalLM):
                    unwrapped_model.save_pretrained(args.output_dir)
                elif not args.disable_lora:
                    from peft import get_peft_model_state_dict

                    state_dict = get_peft_model_state_dict(model, adapter_name="neurocache")
                    torch.save(state_dict, os.path.join(args.output_dir, "pytorch_model.bin"))
                else:
                    state_dict = unwrapped_model.state_dict()
                    torch.save(state_dict, os.path.join(args.output_dir, "pytorch_model.bin"))
    else:
        run_evaluation(
            args, model, eval_dataloader, accelerator, completed_steps, prefix="validation"
        )

    test_dataloader = initialize_dataloader(
        args.per_device_eval_batch_size, task_config, task_config.test_split, accelerator
    )
    run_evaluation(args, model, test_dataloader, accelerator, completed_steps, prefix="test")


if __name__ == "__main__":
    main()
