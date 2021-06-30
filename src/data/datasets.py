import json
import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from src.data.formatting import (
    Triple,
    Entity,
    RelationType,
    Example,
    # TextFormat,
    GraphFormat,
)
from src.utils import camel_case_to_natural_text


class Seq2seqDataset(Dataset, ABC):
    splits: List[str]
    max_seq_length = 256
    dataset_name: str  # name of the folder after processing

    def __init__(
        self,
        data_dir: Path,
        split: str,
        tokenizer: PreTrainedTokenizer,
    ):
        """
        Base class for both graph-to-text and text-to-graph tasks, in a seq2seq setting

        Args:
            data_dir:
            name: name of the dataset to save the cached files
            split: 'train', 'val' or 'test
            tokenizer:
        """
        self.data_dir = data_dir
        self.tokenizer = tokenizer

        # self.text_format = TextFormat()
        self.graph_format = GraphFormat()

        if not os.path.isfile(data_dir / f"processed/{self.dataset_name}/train.pth"):
            # if not already done, preprocess raw data and save it to disk,
            # for training and evaluation
            self.build_dataset()

        self.examples, self.features, self.unique_graph_ids = torch.load(
            data_dir / f"processed/{self.dataset_name}/{split}.pth"
        )

    def build_dataset(self):
        """
        Load raw data, process it (build the list of examples and of features), and save it to the disk.
        This is called at init, if the cached files are not found.
        """

        logging.info(
            f"[{self.dataset_name}] Processed data not found. "
            f"Loading and processing raw data..."
        )
        os.makedirs(self.data_dir / f"processed/{self.dataset_name}", exist_ok=True)

        for split in self.splits:
            # load raw data of the split
            dataset = self.load_raw_dataset(split)

            examples, unique_graph_ids = self.construct_examples(dataset, split)
            features = self.compute_features(examples)
            torch.save(
                (examples, features, unique_graph_ids),
                self.data_dir / f"processed/{self.dataset_name}/{split}.pth",
            )

            # for evaluation, compute ref text files
            if "train" not in split:
                self.build_references(dataset, split=split)

    @abstractmethod
    def load_raw_dataset(self, split: str):
        """
        Load raw data, either from disk, or from huggingface datasets library.

        Returns
            An object that will be passed to construct_examples()
        """
        pass

    @abstractmethod
    def construct_examples(self, raw_dataset, split: str) -> List[Example]:
        """
        Construct the list of Examples from the raw data
        """
        pass

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
        logging.info("Computing features (format and tokenize graph/text sequences)...")
        # format text and graph into sequences
        text_sentences = []
        graph_sentences = []
        for example in tqdm(examples):
            text_sentences.append(example.text)
            graph_sentences.append(self.graph_format.serialize_graph(example.graph))

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
        for i, (text_ids, att_mask_text, graph_ids, att_mask_graph) in enumerate(
            zip(
                text_tok.input_ids,
                text_tok.attention_mask,
                graph_tok.input_ids,
                graph_tok.attention_mask,
            )
        ):
            features.append(
                {
                    "example_id": i,
                    "text_ids": text_ids.tolist(),
                    "att_mask_text": att_mask_text.tolist(),
                    "graph_ids": graph_ids.tolist(),
                    "att_mask_graph": att_mask_graph.tolist(),
                }
            )

        return features

    @abstractmethod
    def build_references(self, dataset, split: str):
        """
        Construct and save files with reference texts, for g2t evaluation.
        For each split, we make as many files as the max number of lexicalizations
        """
        pass

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


class WebNLG2020(Seq2seqDataset):
    splits = [
        "train",
        "dev",
        "test_all",
        "test_seen",
        "test_unseen_ent",
        "test_unseen_cat",  # unseen entities and categories
    ]
    dataset_name = "webnlg2020"

    def load_raw_dataset(self, split: str):
        split_ = "test" if "test" in split else split
        dataset = load_dataset(
            "web_nlg",
            name="release_v3.0_en",
            split=split_,
            ignore_verifications=True,  # https://github.com/huggingface/datasets/issues/2553
        )

        if "test" in split:
            # take g2t test set (not t2g)
            dataset = dataset.filter(
                lambda example: example["test_category"]
                == "rdf-to-text-generation-test-data-with-refs-en"
            )
            # to compute splits for test seen/unseen datasets, filter and keep only examples in the right category
            path = (
                self.data_dir
                / "raw/webnlg/rdf-to-text-generation-test-instance-types.json"
            )
            with open(path) as f:
                # for each example in test set, which category (seen/unseen)
                test_instance_types = json.load(f)
            dataset = dataset.filter(
                lambda example: self.is_in_test_split(
                    example["eid"], split, test_instance_types
                )
            )

        return dataset

    def construct_examples(self, raw_dataset, split: str):
        logging.info(f"[{split}] Constructing examples...")

        examples = []
        # for webnlg, keep a list of example ids corresponding to unique graphs
        # (which can have multiple lexicalizations, hence multiple examples entries)
        # to be used during g2t evaluation (to make 1 prediction/graph)
        unique_graph_ids = []
        for entry in tqdm(raw_dataset):
            # todo: make sure we process the text correctly, like
            #   - https://github.com/QipengGuo/P2_WebNLG2020/blob/main/main.py
            #   - https://github.com/UKPLab/plms-graph2text/blob/master/webnlg/data/generate_input_webnlg.py

            # graph y
            graph = []
            assert len(entry["modified_triple_sets"]["mtriple_set"]) == 1
            raw_triples = entry["modified_triple_sets"]["mtriple_set"][0]
            for raw_triple in raw_triples:
                e1, rel, e2 = raw_triple.split(" | ")
                e1 = e1.replace("_", " ")
                e2 = e2.replace("_", " ")
                rel_natural = camel_case_to_natural_text(rel)
                graph += [
                    Triple(Entity(e1), RelationType(rel, rel_natural), Entity(e2))
                ]
            # id of the first example with this graph
            unique_graph_ids.append(len(examples))
            assert len(entry["lex"]["text"]) > 0

            for text in entry["lex"]["text"]:
                # text x (potentially multiple lexicalizations for the same graph)
                examples.append(Example(text=text, graph=graph))

        logging.info(f"[{split}] unique graphs: {len(unique_graph_ids)}")
        return examples, unique_graph_ids

    def build_references(self, raw_dataset, split: str):
        # use webnlg2020 challenge code to process raw data (Benchmark object)
        # into .txt files for eval
        logging.info(f"[{split}] Building reference files...")
        ref_dir = self.data_dir / f"processed/{self.dataset_name}/ref"
        os.makedirs(ref_dir, exist_ok=True)

        # create reference text files without any processing, like
        # https://gitlab.com/webnlg/corpus-reader/-/blob/master/generate_references.py
        target_out = []
        for entry in raw_dataset:
            entry_refs = []
            for lex in entry["lex"]["text"]:
                entry_refs.append(lex)
            target_out.append(entry_refs)
        # the list with max elements
        max_refs = sorted(target_out, key=len)[-1]
        # write references files
        for j in range(0, len(max_refs)):
            out = []
            for entry_refs in target_out:
                try:
                    out.append(f"{entry_refs[j]}\n")
                except IndexError:
                    out.append("\n")
            # for the last line, don't consider the '\n', to avoid putting an additional
            # empty line in the reference files
            out[-1] = out[-1][:-1]
            with open(ref_dir / f"{split}_{str(j)}.txt", "w+", encoding="utf-8") as f:
                f.write("".join(out))

    def is_in_test_split(self, example_id, split, test_instance_types):
        """
        Return True if the example belongs to this test split
        """
        if split == "test_all":
            return True
        types_to_split = {
            "type1": "test_seen",
            "type2": "test_unseen_ent",
            "type3": "test_unseen_cat",
        }
        example_type = test_instance_types[example_id]
        example_split = types_to_split[example_type]
        return example_split == split
