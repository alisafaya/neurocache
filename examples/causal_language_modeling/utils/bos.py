import tensorflow.compat.v2 as tf


def _prepend_to_innermost_axis(tensor: tf.Tensor, scalar: tf.Tensor) -> tf.Tensor:
    """Appends `scalar` to each slice in the innermost axis of `tensor`.
    >>> _prepend_to_innermost_axis([1, 2, 3], -1)
    [-1, 1, 2, 3]
    >>> _prepend_to_innermost_axis([[1, 2], [3, 4]], -1)
    [[-1, 1, 2], [-1, 3, 4]]
    >>> _prepend_to_innermost_axis(tf.ragged.constant([[1, 2], [3]]), -1)
    [[-1, 1, 2], [-1, 3]]
    Args:
      tensor: The tensor that should have a value appended.
      scalar: The value to append.
    Returns:
      A copy of `tensor` with `scalar` appended to each slice along
      the innermost axis.
    """
    if isinstance(tensor, tf.RaggedTensor):
        if tensor.shape.rank > 2:
            return tensor.with_values(_prepend_to_innermost_axis(tensor.values, scalar))
        else:
            return tf.concat([tf.fill([tensor.nrows(), 1], scalar), tensor], axis=1)
    else:
        ndims = tf.rank(tensor)
        paddings = tf.concat(
            [tf.zeros((ndims - 1, 2), dtype=tf.int32), tf.constant([[1, 0]])], axis=0
        )
        return tf.pad(tensor, paddings=paddings, constant_values=scalar)


def prepend_bos(
    dataset: tf.data.Dataset,
    output_features,
) -> tf.data.Dataset:
    """Prepends BOS to output feature token sequences with `add_bos` set to True.
    Respects the `add_bos` field of the seqio.Features in `output_features`.
    Args:
      dataset: a tf.data.Dataset of tokenized examples to preprocess.
      output_features: a mapping of output feature names to Feature objects.
    Returns:
      a tf.data.Dataset of tokenized examples with EOS added to specified output
      features.
    """

    def _maybe_add_bos(key: str, value: tf.Tensor) -> tf.Tensor:
        if key not in output_features:
            return value
        else:
            # Here we use unk_id as bos_id, since there is no
            # bos_id functionality implemented in seqio.
            # Note that we use a unigram tokenizer with byte
            # fallback, hence there is no unknown tokens.
            bos_id = output_features[key].vocabulary.unk_id
            return _prepend_to_innermost_axis(value, bos_id)

    return dataset.map(
        lambda ex: {k: _maybe_add_bos(k, v) for k, v in ex.items()},
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
