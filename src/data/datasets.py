import json
import logging
import os
import re
from collections import Counter
from pathlib import Path
from typing import List, Tuple, Generator

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm
from transformers import PreTrainedTokenizer, default_data_collator

from src.data.formatting import (
    Relation,
    Entity,
    RelationType,
    Example,
    InputFormat,
    OutputFormat,
)
from src.utils import camel_case_to_natural_text, get_precision_recall_f1


class WebNLG(Dataset):
    raw_data_paths = {
        "train": "raw/webnlg/train.json",
        "val": "raw/webnlg/dev.json",
        "test": "raw/webnlg/test.json",
    }
    max_input_length = 256
    max_output_length = 256

    def __init__(
        self,
        data_dir: Path,
        split: str,
        tokenizer: PreTrainedTokenizer,
        limit_samples=-1,
    ):
        """

        Args:
            data_dir:
            split: 'train', 'val' or 'test
            tokenizer:
        """
        self.tokenizer = tokenizer
        self.split = split
        self.limit_samples = limit_samples

        self.input_format = InputFormat()
        self.output_format = OutputFormat()

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
        input_sentences = [
            self.input_format.format_input(example.text) for example in examples
        ]
        output_sentences = [
            self.output_format.format_output(example.graph) for example in examples
        ]

        input_tok = self.tokenizer.batch_encode_plus(
            input_sentences,
            max_length=self.max_input_length,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )
        self._warn_max_sequence_length(self.max_input_length, input_sentences, "input")

        output_tok = self.tokenizer.batch_encode_plus(
            output_sentences,
            max_length=self.max_output_length,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )
        self._warn_max_sequence_length(
            self.max_output_length, output_sentences, "output"
        )

        assert (
            input_tok.input_ids.size(0) == output_tok.input_ids.size(0) == len(examples)
        )

        features = []
        for input_ids, att_mask, label_ids in zip(
            input_tok.input_ids, input_tok.attention_mask, output_tok.input_ids
        ):
            features.append(
                {
                    "input_ids": input_ids.tolist(),
                    "attention_mask": att_mask.tolist(),
                    "label_ids": label_ids.tolist(),
                }
            )

        return features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index]

    def get_example(self, index):
        return self.examples[index]

    def evaluate_example(self, example: Example, output_sentence: str):
        (
            predicted_entities,
            predicted_relations,
            wrong_format,
        ) = self.output_format.run_inference(output_sentence)

        # load ground truth entities and relations (as str and tuple of str)
        gt_entities = set()
        gt_relations = set()
        for relation in example.graph:
            gt_entities |= {relation.head.text, relation.tail.text}
            gt_relations |= {relation.type.natural}

        # compute correct entities
        correct_entities = predicted_entities & gt_entities
        # compute correct relations
        correct_relations = predicted_relations & gt_relations

        assert len(correct_entities) <= len(predicted_entities)
        assert len(correct_entities) <= len(gt_entities)
        assert len(correct_relations) <= len(predicted_relations)
        assert len(correct_relations) <= len(gt_relations)

        res = Counter(
            {
                "num_sentences": 1,
                "wrong_reconstructions": 1 if wrong_format else 0,
                "gt_entities": len(gt_entities),
                "predicted_entities": len(predicted_entities),
                "correct_entities": len(correct_entities),
                "gt_relations": len(gt_relations),
                "predicted_relations": len(predicted_relations),
                "correct_relations": len(correct_relations),
            }
        )
        # todo: add information about each entity/relation type to compute macro f1 scores?

        return res

    def evaluate_dataset(
        self,
        model,
        device,
        batch_size: int,
        num_beams: int,
    ):
        """
        Evaluate model on this dataset.
        """
        results = Counter()

        logs = ""
        for i, (example, output_sentence, input_sentence, label_sentence) in enumerate(
            self.generate_output_sentences(model, device, batch_size, num_beams)
        ):
            new_result = self.evaluate_example(
                example=example,
                output_sentence=output_sentence,
            )
            results += new_result
            logs += (
                f"[{i}] input / output / label\n"
                f"{input_sentence}\n"
                f"{output_sentence}\n"
                f"{label_sentence}\n"
            )

        entity_precision, entity_recall, entity_f1 = get_precision_recall_f1(
            num_correct=results["correct_entities"],
            num_predicted=results["predicted_entities"],
            num_gt=results["gt_entities"],
        )
        relation_precision, relation_recall, relation_f1 = get_precision_recall_f1(
            num_correct=results["correct_relations"],
            num_predicted=results["predicted_relations"],
            num_gt=results["gt_relations"],
        )

        res = {
            f"{self.split}_wrong_reconstruction": results["wrong_reconstructions"]
            / results["num_sentences"],
            f"{self.split}_label_error": results["label_error"]
            / results["num_sentences"],
            f"{self.split}_entity_error": results["entity_error"]
            / results["num_sentences"],
            f"{self.split}_format_error": results["format_error"]
            / results["num_sentences"],
            f"{self.split}_entity_precision": entity_precision,
            f"{self.split}_entity_recall": entity_recall,
            f"{self.split}_entity_f1": entity_f1,
            f"{self.split}_relation_precision": relation_precision,
            f"{self.split}_relation_recall": relation_recall,
            f"{self.split}_relation_f1": relation_f1,
        }

        return res, logs

    def generate_output_sentences(
        self,
        model,
        device,
        batch_size: int,
        num_beams: int,
    ) -> Generator[Tuple[Example, str], None, None]:
        """
        Generate pairs (example, output_sentence) for evaluation on this dataset.
        """

        # to speed up validation, only consider a few samples
        dataset = Subset(self, range(self.limit_samples)) if self.limit_samples > 0 else self
        eval_data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=default_data_collator,
        )

        logging.info(f"Generating sentences on dataset {self.split}...")
        for i, inputs in tqdm(enumerate(eval_data_loader), total=len(eval_data_loader)):
            predictions = model.generate(
                inputs["input_ids"].to(device),
                max_length=self.max_output_length,
                num_beams=num_beams,
            )

            for j, (input_ids, label_ids, prediction) in enumerate(
                zip(inputs["input_ids"], inputs["labels"], predictions)
            ):
                current_id = i * batch_size + j
                example = self.get_example(current_id)
                output_sentence = self.tokenizer.decode(
                    prediction,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                input_sentence = self.tokenizer.decode(
                    input_ids,
                    skip_special_tokens=True,
                )
                label_sentence = self.tokenizer.decode(
                    label_ids,
                    skip_special_tokens=True,
                )

                yield example, output_sentence, input_sentence, label_sentence

    def _warn_max_sequence_length(
        self, max_sequence_length: int, sentences: List[str], name: str
    ):
        max_length_needed = max(len(self.tokenizer.tokenize(x)) for x in sentences)
        if max_length_needed > max_sequence_length:
            logging.warning(
                f"Max sequence length is {max_sequence_length} but the longest {name} sequence is "
                f"{max_length_needed} long"
            )
