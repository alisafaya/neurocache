"""Add Tasks to seqio registry."""

import functools
from typing import Mapping, Optional, Set

import seqio
import tensorflow as tf

from data_utils import bos


def rekey_articles(
    ds: tf.data.Dataset, rekey: Mapping[str, str], keep: Optional[Set[str]] = None
) -> tf.data.Dataset:
    """Rekey the articles in ds.

    Fields in rekey will be renamed, field in keep will be kept, others will
    be discarded.  E.g., For PG19:

      rekey_article(ds,
                    rekey={"book_text": "targets"},
                    keep={"book_title", "book_id"})
    Args:
      ds: The dataset to rekey.
      rekey: Dictionary which contains fields to rename.
      keep: Set of fields to keep.

    Returns:
      A rekeyed dataset.
    """

    def rekey_fn(article):
        result_dict = {}
        for k, v in article.items():
            if k in rekey:
                result_dict[rekey[k]] = v
            elif k in keep:
                result_dict[k] = v
        return result_dict

    return ds.map(rekey_fn)


def define_pretraining_task(tfds_name: str, tfds_version: str, vocab: seqio.Vocabulary):
    seqio.TaskRegistry.add(
        f"{tfds_name}:{tfds_version}",
        seqio.TfdsDataSource(tfds_name=f"{tfds_name}:{tfds_version}"),
        preprocessors=[
            functools.partial(rekey_articles, rekey={"text": "targets"}, keep={"subset"}),
            seqio.preprocessors.tokenize,
            seqio.preprocessors.append_eos,
            bos.prepend_bos,
        ],
        output_features={
            "targets": seqio.Feature(vocab, add_eos=True, dtype=tf.int32),
        },
    )
