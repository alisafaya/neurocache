"""
Extract Statistics from a Dataset such as the Document Length and the Number of Tokens.
"""

import logging

import seqio
import tensorflow.compat.v2 as tf
from tqdm import tqdm
from utils import data_utils as dutils
from utils.args import parse_args


# Make sure that tensorflow is not reserving GPUs.
tf.config.experimental.set_visible_devices([], "GPU")


def main():
    args = parse_args()

    print(args)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # Set up seqio tasks
    if args.tokenizer_name is not None:
        dutils.setup_tasks(args.tokenizer_name)
    else:
        dutils.setup_tasks(args.model_name_or_path)

    task = seqio.get_mixture_or_task(args.dataset_name)

    # Get the dataset.
    ds = task.get_dataset(
        split=args.train_split, sequence_length={"targets": None}, shuffle=False, num_epochs=1
    )

    # Get the document length and the number of tokens.
    document_lengths = []
    token_counts = []
    for example in tqdm(ds, desc="Extracting Document Lengths"):
        token_counts.append(example["targets"].shape[0])
        document_lengths.append(len(example["targets_pretokenized"].numpy()))

    with open(args.dataset_name.replace(":", ".") + "_document_lengths.csv", "w") as f:
        f.write("chars,tokens\n")
        for document_length, token_count in zip(document_lengths, token_counts):
            f.write(f"{document_length},{token_count}\n")


if __name__ == "__main__":
    main()
