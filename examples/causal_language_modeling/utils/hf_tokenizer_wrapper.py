from typing import Optional, Sequence

import numpy as np
import tensorflow.compat.v2 as tf
from seqio import Vocabulary
from transformers import AutoTokenizer


class HFVocabulary(Vocabulary):
    """SeqIO Vocabulary based on a HuggingFace tokenizer."""

    def __init__(self, tokenizer_name: str):
        """Vocabulary constructor.

        Args:
            pretrained_tokenizer: The name of the pretrained tokenizer.
            extra_ids: The number of extra IDs to reserve.
        """
        super().__init__(extra_ids=0)
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    @property
    def eos_id(self) -> Optional[int]:
        return self._tokenizer.eos_token_id

    @property
    def pad_id(self) -> int:
        return self._tokenizer.pad_token_id

    @property
    def unk_id(self) -> Optional[int]:
        # Since we use unk as bos, we prioritize bos over unk,
        if self._tokenizer.bos_token_id is not None:
            return self._tokenizer.bos_token_id
        elif self._tokenizer.unk_token_id is not None:
            return self._tokenizer.unk_token_id

    @property
    def _base_vocab_size(self) -> int:
        return self._tokenizer.vocab_size

    def _encode(self, string: str) -> Sequence[int]:
        ids = self._tokenizer.encode(
            string,
            truncation=False,
            return_tensors="np",
            max_length=None,
            padding=False,
            add_special_tokens=False,
        )
        return ids

    def _decode(self, ids):
        return self._tokenizer.decode(ids)

    def _encode_tf(self, string: tf.Tensor):
        def encode_fn(s):
            if isinstance(s, bytes):
                s = s.decode("utf-8")
            ids = self._encode(s)
            ids = np.array(ids, dtype=np.int32)[0]
            return ids

        ids = tf.numpy_function(encode_fn, [string], Tout=tf.int32)
        ids.set_shape([None])
        return ids

    def _decode_tf(self, ids: tf.Tensor):
        def decode_fn(ids):
            ids = ids.numpy().tolist()
            s = self._decode(ids)
            return s

        s = tf.numpy_function(decode_fn, [ids], Tout=tf.string)
        s.set_shape([None])
        return s
