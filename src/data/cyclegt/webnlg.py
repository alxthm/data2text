import json
import json
import os
from pathlib import Path

import torch
from transformers import BertTokenizer

from src.data.cyclegt.shared import scan_data, Example, CycleCVAECollator, CycleCVAEDataset


def prepare_data(data_dir, device, is_supervised: bool):
    """
    If processed data is not found in `data/processed`:
        - load the files in `data/raw`
        - build and save the vocabulary (for text, entities and relations)
        - build and save the processed data (for train, val, test)

    Args:
        data_dir:
        device:
        is_supervised:

    Returns:
        train, val, test datasets from processed data.
    """
    dataset_train_path = data_dir / "processed/webnlg/train.data"
    dataset_val_path = data_dir / "processed/webnlg/val.data"
    dataset_test_path = data_dir / "processed/webnlg/test.data"
    vocab_path = data_dir / "processed/webnlg/vocab.data"

    # if necessary, build vocabulary and save processed datasets to disk
    if not os.path.isfile(data_dir / "processed/webnlg/train.data"):
        os.makedirs(data_dir / "processed/webnlg", exist_ok=True)

        # load raw data files
        with open(data_dir / "raw/webnlg/train.json", "r") as f:
            train_raw = json.load(f)
        with open(data_dir / "raw/webnlg/dev.json", "r") as f:
            val_raw = json.load(f)
        with open(data_dir / "raw/webnlg/test.json", "r") as f:
            test_raw = json.load(f)

        # remove top 5% longer sequences in train
        max_len = sorted([len(x["text"].split()) for x in train_raw])[
            int(0.95 * len(train_raw))
        ]
        train_raw = [x for x in train_raw if len(x["text"].split()) < max_len]
        print(f"max sequence length in training: {max_len}")

        # download pre-trained tokenizer if not already available
        BertTokenizer.from_pretrained("bert-base-uncased")

        # build and save vocab
        print("Building vocab...")
        vocab = scan_data(train_raw)
        vocab = scan_data(val_raw, vocab)
        vocab = scan_data(test_raw, vocab, is_test=True)
        for k, v in vocab.items():
            v.build()
            print(f"Vocab {k}: size {len(v)}, and in test set: {len(v.sp)}")
        torch.save(vocab, vocab_path)

        # build and save datasets
        train_data = [Example(x, vocab).get() for x in train_raw]
        val_data = [Example(x, vocab).get() for x in val_raw]
        test_data = [Example(x, vocab).get() for x in test_raw]
        torch.save(train_data, dataset_train_path)
        torch.save(val_data, dataset_val_path)
        torch.save(test_data, dataset_test_path)

    if is_supervised:
        dataset_train_t2g = CycleCVAEDataset(dataset_train_path, mode="both")
        dataset_train_g2t = CycleCVAEDataset(dataset_train_path, mode="both")
    else:
        dataset_train_t2g = CycleCVAEDataset(dataset_train_path, mode="graph_only")
        dataset_train_g2t = CycleCVAEDataset(dataset_train_path, mode="text_only")
    dataset_val = CycleCVAEDataset(dataset_val_path, mode="both")
    dataset_test = CycleCVAEDataset(dataset_test_path, mode="both")

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    vocab = torch.load(vocab_path)
    text_vocab = vocab["text"]
    ent_vocab = vocab["entity"]
    rel_vocab = vocab["relation"]
    collate_fn_train = CycleCVAECollator(
        ent_vocab=ent_vocab,
        text_vocab=text_vocab,
        tokenizer=tokenizer,
        device=device,
        is_eval=False,
    )
    collate_fn_eval = CycleCVAECollator(
        ent_vocab=ent_vocab,
        text_vocab=text_vocab,
        tokenizer=tokenizer,
        device=device,
        is_eval=True,
    )

    return (
        dataset_train_t2g,
        dataset_train_g2t,
        dataset_val,
        dataset_test,
        vocab,
        collate_fn_train,
        collate_fn_eval,
    )


if __name__ == "__main__":
    data_dir = Path(__file__).parents[2].resolve() / "data"
    with open(data_dir / "raw/webnlg/train.json", "r") as f:
        train_raw = json.load(f)
    with open(data_dir / "raw/webnlg/dev.json", "r") as f:
        val_raw = json.load(f)
    with open(data_dir / "raw/webnlg/test.json", "r") as f:
        test_raw = json.load(f)

    # remove top 5% longer sequences in train
    max_len = sorted([len(x["text"].split()) for x in train_raw])[
        int(0.95 * len(train_raw))
    ]
    train_raw = [x for x in train_raw if len(x["text"].split()) < max_len]
    print(f"max sequence length in training: {max_len}")

    # download pre-trained tokenizer if not already available
    BertTokenizer.from_pretrained("bert-base-uncased")

    # build and save vocab
    print("Building vocab...")
    vocab = scan_data(train_raw)
    vocab = scan_data(val_raw, vocab)
    vocab = scan_data(test_raw, vocab, is_test=True)
    for k, v in vocab.items():
        v.build()
        print(f"Vocab {k}: size {len(v)}, and in test set: {len(v.sp)}")

    # build and save datasets
    train_data = [Example(x, vocab).get() for x in train_raw]
    val_data = [Example(x, vocab).get() for x in val_raw]
    test_data = [Example(x, vocab).get() for x in test_raw]
