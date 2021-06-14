import json
import os
from glob import glob
from pathlib import Path

import random

import torch
from tqdm import tqdm
from transformers import BertTokenizer

from src.data._old_cyclegt.shared import (
    CycleCVAEDataset,
    CycleCVAECollator,
    Example,
    scan_data,
)


def load_raw_dataset(path: Path):
    print(f"Loading raw dataset {path.name}")
    # load separated data files into one big list of samples
    files = glob(str(path / "*.json"))
    dataset = []
    for file in tqdm(files):
        with open(file) as f:
            data = json.load(f)
            # change the data to match WebNLG format
            data_webnlg = []
            for d in data:
                entities = [[e] for e in d["entities"]]
                # replace empty entities (from original list) by a blank character
                entities = [e if e != [""] else ["_"] for e in entities]

                relations = []
                for (e1, r, e2) in d["graph"]:
                    # replace empty entities (in graph) by a blank character
                    if e1 == "":
                        e1 = "_"
                    if e2 == "":
                        e2 = "_"
                    relations.append([[e1], r, [e2]])
                    # by default, GenWiki only has entities from the text in 'entities'
                    # -> add the ones present in the graph (and not the text) as well,
                    # to match WebNLG
                    if e1 not in entities:
                        entities.append([e1])
                    if e2 not in entities:
                        entities.append([e2])
                data_webnlg.append(
                    {
                        "relations": relations,
                        "text": d["text"],
                        "entities": entities,
                    }
                )

            dataset += data_webnlg

    return dataset


def read(project_dir: Path):
    raw_data_dir = project_dir / f"data/raw/genwiki"

    all_data = {}
    for name in ["train/full", "train/fine", "test"]:
        all_data[name] = load_raw_dataset(raw_data_dir / name)

        print(
            f"[Info] There are {len(all_data[name])} samples in the GenWiki-{name} dataset"
        )

    return all_data


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
    dataset_train_path = data_dir / "processed/genwiki/train-fine.data"
    dataset_val_path = data_dir / "processed/genwiki/val-fine.data"
    dataset_test_path = data_dir / "processed/genwiki/test.data"
    vocab_path = data_dir / "processed/genwiki/vocab.data"

    # if necessary, build vocabulary and save processed datasets to disk
    if not os.path.isfile(data_dir / "processed/genwiki/test.data"):
        os.makedirs(data_dir / "processed/genwiki", exist_ok=True)
        random.seed(0)

        # load raw data files
        test_raw = load_raw_dataset(data_dir / "raw/genwiki/test")
        train_fine_raw = load_raw_dataset(data_dir / "raw/genwiki/train/fine")
        # shuffle and split train dataset into train and val
        random.shuffle(train_fine_raw)
        val_dataset_size = 10000
        val_raw = train_fine_raw[:val_dataset_size]
        train_raw = train_fine_raw[val_dataset_size:]

        # remove top 5% longer sequences in train
        # max_len = sorted([len(x["text"].split()) for x in train_raw])[
        #     int(0.95 * len(train_raw))
        # ]
        # train_raw = [x for x in train_raw if len(x["text"].split()) < max_len]

        # download pre-trained tokenizer if not already available
        BertTokenizer.from_pretrained("bert-base-uncased")

        # build and save vocab
        print("Building vocab (train/val/test)...")
        vocab = scan_data(train_raw)
        vocab = scan_data(val_raw, vocab)
        vocab = scan_data(test_raw, vocab, is_test=True)
        for k, v in vocab.items():
            v.build()
            print(f"Vocab {k}: size {len(v)}, and in test set {len(v.sp)}")
        torch.save(vocab, vocab_path)

        # build and save datasets
        print("Building datasets (train/val/test)...")
        train_data = [Example(x, vocab).get() for x in tqdm(train_raw)]
        val_data = [Example(x, vocab).get() for x in tqdm(val_raw)]
        test_data = [Example(x, vocab).get() for x in tqdm(test_raw)]
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

    # todo: assert it's the right length (full: 1336766, fine: 757152, test: 1000)

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
    project_dir = Path(__file__).parents[2].resolve()
    # read(project_dir)
    prepare_data(project_dir / "data", device=torch.device("cpu"), is_supervised=True)
