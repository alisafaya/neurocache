from dataclasses import dataclass

import torch
from absl import logging
from torch.utils.data import IterableDataset

from data_utils.hf_tokenizer_wrapper import HFVocabulary
from data_utils.seqio_task import define_pretraining_task
from data_utils.text_dataset import load_text_dataset


@dataclass
class TrainingTaskConfig:
    """Configuration hyperparameters for sequence-to-sequence tasks."""

    model_name_or_path: str = "facebook/opt-350m"  # Pretrained model name or path
    dataset_name: str = "long_pile:1.1.0"
    train_split: str = "train"
    eval_split: str = "validation"
    test_split: str = "test"
    sequence_length: int = 512  # Maximum sequence length of chunks.
    per_device_batch_size: int = 1  # per device batch size
    max_train_steps: int = 100  # Number of training steps.
    max_eval_steps: int = 10  # Number of evaluation steps.
    seed: int = 1234567  # Random seed for initialization.

    @classmethod
    def from_args(cls, args):
        """Build a new TrainingTaskConfig from argparse arguments."""
        args = vars(args)
        return cls.from_dict(args)

    @classmethod
    def from_dict(cls, dict):
        """Build a new TrainingTaskConfig from a dictionary."""
        # remove the keys that are not in the dataclass
        dict = dict.copy()
        for key in list(dict.keys()):
            if key not in cls.__dataclass_fields__.keys() or dict[key] is None:
                del dict[key]
        return cls(**dict)


class LongTextDataset(IterableDataset):
    def __init__(self, task_config, split, total_bsz, **kwargs):
        self.ds, self.vocab = load_text_dataset(
            split=split,
            sequential=True,
            name=task_config.dataset_name,
            sequence_length=task_config.sequence_length + 1,
            batch_size=total_bsz,
            seed=task_config.seed,
            **kwargs,
        )
        self.ds_iter = None

    def __iter__(self):
        self.ds_iter = self.ds.as_numpy_iterator()
        return self

    def __next__(self):
        try:
            data = next(self.ds_iter)
            data = {
                "input_ids": torch.LongTensor(data["targets"][:, :-1].copy()),
                "attention_mask": torch.LongTensor(data["loss_mask"][:, :-1].copy()),
                "labels": torch.LongTensor(data["targets"][:, 1:].copy()),
                "start_of_sequence": torch.BoolTensor(data["start_of_sequence"].copy()),
                "epochs": torch.LongTensor(data["epoch"].copy()),
            }
            return data
        except StopIteration:
            self.ds_iter = self.ds.as_numpy_iterator()
            return next(self)


def setup_tasks(tokenizer_path: str):
    """Setup the pretraining tasks."""
    define_pretraining_task("long_pile", "1.1.0", HFVocabulary(tokenizer_path))


def main(args):
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM

    from neurocache import NeurocacheModelForCausalLM, OnDeviceCacheConfig

    logging.info("Setting up the pretraining tasks.")
    setup_tasks(args.model_name_or_path)

    logging.info("Setting up the data pipeline.")
    task_config = TrainingTaskConfig.from_args(args)

    logging.info("Starting the training loop.")
    train_dataset = LongTextDataset(
        task_config, task_config.eval_split, args.per_device_batch_size
    )

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    neurocache_config = OnDeviceCacheConfig(
        cache_layers=[9],
        cache_size=4096,
        cache_dtype="float16",
        attention_layers=[9, 10, 11],
        topk=8,
        compression_factor=4,
    )
    model = NeurocacheModelForCausalLM(model, neurocache_config)

    model.eval()
    model = model.to("cuda")
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            total_loss = 0
            for i, batch in tqdm(enumerate(train_dataset)):
                batch.pop("epochs")
                batch = {k: v.to("cuda") for k, v in batch.items()}
                target_ids = batch.pop("labels")
                outputs = model(**batch)
                loss = torch.nn.functional.cross_entropy(
                    outputs.logits.view(-1, outputs.logits.size(-1)).to(torch.float32),
                    target_ids.view(-1),
                    ignore_index=0,
                )
                total_loss += loss.detach().item()

                if i % 10 == 0 and i > 0:
                    print(f"Step {i}: {total_loss / (i + 1)}")

                if i >= args.max_eval_steps:
                    break

        print(f"Final loss: {total_loss / (i + 1)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="gpt2")
    parser.add_argument("--dataset_name", type=str, default="long_pile:1.1.0")
    parser.add_argument("--sequence_length", type=int, default=512)
    parser.add_argument("--per_device_batch_size", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=100)
    parser.add_argument("--max_eval_steps", type=int, default=20)

    args = parser.parse_args()
    main(args)
