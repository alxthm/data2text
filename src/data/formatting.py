from typing import List, Tuple

import torch
from dataclasses import dataclass
from transformers import PreTrainedTokenizer

GENERATE_TEXT_TOKEN = "[GENERATE_TEXT]"
GENERATE_GRAPH_TOKEN = "[GENERATE_GRAPH]"
STYLE_TOKEN = "[STYLE]"


@dataclass
class Entity:
    text: str


@dataclass
class RelationType:
    short: str
    natural: str  # string to use in input/output sequences, more readable


@dataclass
class Triple:
    head: Entity
    rel: RelationType
    tail: Entity

    def __repr__(self):
        return f"({self.head.text} -> {self.rel.short} -> {self.tail.text})"

    def to_tuple(self):
        return self.head.text, self.rel.natural, self.tail.text


@dataclass
class Example:
    text: str  # plain text sentence
    graph: List[Triple]


# class TextFormat:
#     prefix_t2g = "text to graph : "
#
#     def format_input(self, text: str):
#         return self.prefix_t2g + text


class GraphFormat:
    HEAD_TOKEN = "[HEAD]"
    TYPE_TOKEN = "[TYPE]"
    TAIL_TOKEN = "[TAIL]"
    BLANK_TOKEN = "[BLANK]"

    @staticmethod
    def serialize_graph(graph: List[Triple]) -> str:
        """
        Format graph (list of relations) into a string

        Examples
            for a graph with only one relation:
            '[HEAD] Abilene , Texas [TYPE] city served [TAIL] Abilene Regional Airport'

        """
        seralized_graphs = []
        for triple in graph:
            if triple == GraphFormat.BLANK_TOKEN:
                # make it possible to replace entire triplets with a [BLANK] token
                seralized_graphs.append(triple)
            else:
                seralized_graphs.append(
                    " ".join(
                        [
                            GraphFormat.HEAD_TOKEN,
                            triple.head.text,
                            GraphFormat.TYPE_TOKEN,
                            triple.rel.natural,
                            GraphFormat.TAIL_TOKEN,
                            triple.tail.text,
                        ]
                    )
                )
        return " ".join(seralized_graphs)

    @staticmethod
    def extract_raw_graph(output_sentence: str) -> Tuple[set, set, bool]:
        """
        Parse raw output sentence, extract entities and relations

        Returns:
            Raw set of entities (str) and triples (tuple of str), and whether
            there was a parsing error at some point

        Examples:
            output_sentence:
                '[HEAD] Abilene , Texas [TYPE] city served [TAIL] Abilene Regional Airport'

            predicted_entities:
                {"Abilene , Texas", "Abilene Regional Airport"}
            predicted_relations:
                {("Abilene , Texas", "city served", "Abilene Regional Airport"),}
            format_error:
                False

        """
        format_error = False
        predicted_entities = set()
        predicted_relations = set()

        # parse_output_sentence
        for relation in output_sentence.split(GraphFormat.HEAD_TOKEN):
            if len(relation) == 0:
                # if '[HEAD]' is at the beginning of the sentence we can obtain an empty str
                continue

            # try splitting head from type and tail
            split_type = relation.split(GraphFormat.TYPE_TOKEN)
            if len(split_type) != 2:
                format_error = True
                continue
            head, type_and_tail = split_type

            # try splitting type and tail
            split_tail = type_and_tail.split(GraphFormat.TAIL_TOKEN)
            if len(split_tail) != 2:
                format_error = True
                continue
            type, tail = split_tail

            e1 = head.strip()
            rel = type.strip()
            e2 = tail.strip()
            predicted_entities.update([e1, e2])
            predicted_relations.add((e1, rel, e2))

        return predicted_entities, predicted_relations, format_error

    @staticmethod
    def split_triples(sequence) -> List[str]:
        """
        Split an input sequence into a list of triples (still as strings)

        Examples
            >>> sequence = "[HEAD] 20 Fenchurch Street [TYPE] location [TAIL] London [HEAD] London [TYPE] leader title [TAIL] European Parliament"
            >>> GraphFormat.split_triples(sequence)
            ["[HEAD] 20 Fenchurch Street [TYPE] location [TAIL] London",
             "[HEAD] London [TYPE] leader title [TAIL] European Parliament"]
        """
        triples = sequence.split(GraphFormat.HEAD_TOKEN)
        # don't consider the empty string at the beginning, given by split()
        assert len(triples[0]) == 0
        triples = triples[1:]
        # add the HEAD token back at the beginning of the triple str
        triples = [f"{GraphFormat.HEAD_TOKEN} {t.strip()}" for t in triples]
        return triples


def add_target_prefix(
    input_ids: torch.Tensor,
    target: str,
    tokenizer: PreTrainedTokenizer,
    max_seq_len: int,
):
    # this effectively adds 4 tokens at the beginning of the sequence, but keeps a length
    # of max_seq_len -> some tokens might be discarded during the process
    if target == "text":
        prefix = "Generate text: "
    elif target == "graph":
        prefix = "Generate graph: "
    else:
        return ValueError

    # decode input id sequence (batch of tokenized inputs), and add prefix
    input_str = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
    input_str = [prefix + s for s in input_str]
    # encode back
    batch_encoding = tokenizer(
        input_str,
        max_length=max_seq_len,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
    )
    batch_encoding = batch_encoding.to(input_ids.device)
    return batch_encoding.input_ids


def add_style_prefix(input_ids: torch.Tensor, tokenizer: PreTrainedTokenizer):
    """
    Take input_ids, shift it to the right and add the [STYLE] special token at the beginning.
    Inspired from T5 `_shift_right` method
    """
    style_token_id = tokenizer.convert_tokens_to_ids(STYLE_TOKEN)

    # shift inputs to the right
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
    shifted_input_ids[..., 0] = style_token_id
    return shifted_input_ids
