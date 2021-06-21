import json
import logging
import os
import re
from pathlib import Path
from typing import List

import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from src.data.formatting import (
    Relation,
    Entity,
    RelationType,
    Example,
    # TextFormat,
    GraphFormat,
)
from src.utils import camel_case_to_natural_text


class WebNLG(Dataset):
    raw_data_paths = {
        "train": "raw/webnlg/train.json",
        "val": "raw/webnlg/dev.json",
        "test": "raw/webnlg/test.json",
    }
    max_seq_length = 256

    def __init__(
        self,
        data_dir: Path,
        split: str,
        tokenizer: PreTrainedTokenizer,
    ):
        """

        Args:
            data_dir:
            split: 'train', 'val' or 'test
            tokenizer:
        """
        self.tokenizer = tokenizer
        self.split = split

        # self.text_format = TextFormat()
        self.graph_format = GraphFormat()

        if not os.path.isfile(data_dir / "processed/webnlg_seq2seq/train.pth"):
            # if not already done, preprocess raw data and save it to disk
            logging.info(
                "Processed data not found.\nLoading and processing raw data..."
            )
            os.makedirs(data_dir / "processed/webnlg_seq2seq", exist_ok=True)

            # todo: use the real WebNLG dataset as raw data
            #   https://gitlab.com/shimorina/webnlg-dataset/-/tree/master/release_v3.0
            for split, relative_path in self.raw_data_paths.items():
                logging.info(f"Processing split {split}...")
                # load raw data files
                with open(data_dir / relative_path, "r") as f:
                    raw_data = json.load(f)

                examples = self.construct_examples(raw_data)
                features = self.compute_features(examples)

                torch.save(
                    (examples, features),
                    data_dir / f"processed/webnlg_seq2seq/{split}.pth",
                )

        self.examples, self.features = torch.load(
            data_dir / f"processed/webnlg_seq2seq/{split}.pth"
        )

    def construct_examples(self, raw_data: List[dict]):
        # construct the list of examples
        examples = []
        for x in tqdm(raw_data):
            # graph y
            graph = []
            for e1, t, e2 in x["relations"]:
                t_natural = camel_case_to_natural_text(t)
                graph += [
                    Relation(
                        Entity(" ".join(e1)),
                        RelationType(t, t_natural),
                        Entity(" ".join(e2)),
                    )
                ]
            # text x
            words = x["text"].split()
            for i, w in enumerate(words):
                if w[:5] == "<ENT_" and w[-1] == ">":
                    # find entity id by removing all non-numerical characters in '<ENT_5>'
                    entity_id = int(re.sub(r"[^0-9]", "", w))
                    # replace '<ENT_5>' by entity full text in the sentence
                    words[i] = " ".join(x["entities"][entity_id])
            text = " ".join(words)
            examples.append(Example(text=text, graph=graph))
        return examples

    def compute_features(self, examples: List[Example]):
        """

        Args:
            examples:

        Returns:
            features: list of (input_ids, att_mask, label_ids) to be used by the seq2seq model

        Examples
            input_ids: tokenized version of
                'text to graph: Abilene , Texas is served by the Abilene Regional Airport .'
            label_ids: tokenized version of
                '[HEAD] Abilene , Texas [TYPE] city served [TAIL] Abilene Regional Airport'

        """
        # format text and graph into sequences
        text_sentences = [example.text for example in examples]
        graph_sentences = [
            self.graph_format.serialize_graph(example.graph) for example in examples
        ]

        text_tok = self.tokenizer.batch_encode_plus(
            text_sentences,
            max_length=self.max_seq_length,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )
        self._warn_max_sequence_length(self.max_seq_length, text_sentences, "input")

        graph_tok = self.tokenizer.batch_encode_plus(
            graph_sentences,
            max_length=self.max_seq_length,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )
        self._warn_max_sequence_length(self.max_seq_length, graph_sentences, "output")

        assert (
            text_tok.input_ids.size(0) == graph_tok.input_ids.size(0) == len(examples)
        )

        features = []
        for text_ids, att_mask_text, graph_ids, att_mask_graph in zip(
            text_tok.input_ids,
            text_tok.attention_mask,
            graph_tok.input_ids,
            graph_tok.attention_mask,
        ):
            features.append(
                {
                    "text_ids": text_ids.tolist(),
                    "att_mask_text": att_mask_text.tolist(),
                    "graph_ids": graph_ids.tolist(),
                    "att_mask_graph": att_mask_graph.tolist(),
                }
            )

        return features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index]

    def get_example(self, index):
        return self.examples[index]

    def _warn_max_sequence_length(
        self, max_sequence_length: int, sentences: List[str], name: str
    ):
        max_length_needed = max(len(self.tokenizer.tokenize(x)) for x in sentences)
        if max_length_needed > max_sequence_length:
            logging.warning(
                f"Max sequence length is {max_sequence_length} but the longest {name} sequence is "
                f"{max_length_needed} long"
            )
