from dataclasses import dataclass
from typing import Iterator

import torch
from seqio import SentencePieceVocabulary
from torch.utils.data import IterableDataset

from utils.hf_tokenizer_wrapper import HFVocabulary
from utils.seqio_task import define_pg19, define_pretraining_task
from utils.text_dataset import load_text_dataset


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
    def __init__(self, task_config, split, total_bsz, skip=0, **kwargs):
        self.ds, self.vocab = load_text_dataset(
            split=split,
            sequential=True,
            name=task_config.dataset_name,
            sequence_length=task_config.sequence_length + 1,
            batch_size=total_bsz,
            seed=task_config.seed,
            **kwargs,
        )
        self.skip = skip

    def __iter__(self):
        if self.skip > 0:
            self.ds_iter = self.ds.skip(self.skip).as_numpy_iterator()
            self.skip = 0
        else:
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


class PlaceholderDataset(IterableDataset):
    def __init__(self, task_config, split, total_bsz, skip=0, **kwargs):
        self.skip = skip
        self.total_bsz = total_bsz
        self.seq_len = task_config.sequence_length

    def __iter__(self) -> Iterator:
        return self

    def __next__(self):
        return {
            "input_ids": torch.LongTensor(self.total_bsz, self.seq_len).fill_(1),
            "attention_mask": torch.LongTensor(self.total_bsz, self.seq_len).fill_(1),
            "labels": torch.LongTensor(self.total_bsz, self.seq_len).fill_(1),
            "start_of_sequence": torch.BoolTensor(self.total_bsz).fill_(False),
            "epochs": torch.LongTensor(self.total_bsz).fill_(0),
        }


def setup_tasks(tokenizer_path: str):
    """Setup the pretraining tasks."""
    vocab_cls = SentencePieceVocabulary if tokenizer_path.endswith(".model") else HFVocabulary
    define_pg19(vocab_cls(tokenizer_path))
    define_pretraining_task("long_pile", "1.0.0", vocab_cls(tokenizer_path))
    define_pretraining_task("long_pile", "1.1.0", vocab_cls(tokenizer_path))
    define_pretraining_task("long_long_pile", "1.0.0", vocab_cls(tokenizer_path))
