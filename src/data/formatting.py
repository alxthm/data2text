from enum import Enum
from typing import List, Tuple

from dataclasses import dataclass

class Mode(Enum):
    t2g = "t2g"
    g2t = "g2t"
    # both_sup and both_unsup to add


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

    def serialize_graph(self, graph: List[Triple]) -> str:
        """
        Format graph (list of relations) into a string

        Examples
            for a graph with only one relation:
            '[HEAD] Abilene , Texas [TYPE] city served [TAIL] Abilene Regional Airport'

        """
        seralized_graphs = []
        for triple in graph:
            seralized_graphs.append(
                " ".join(
                    [
                        self.HEAD_TOKEN,
                        triple.head.text,
                        self.TYPE_TOKEN,
                        triple.rel.natural,
                        self.TAIL_TOKEN,
                        triple.tail.text,
                    ]
                )
            )
        return " ".join(seralized_graphs)

    def extract_raw_graph(self, output_sentence: str) -> Tuple[set, set, bool]:
        """
        Parse raw output sentence, extract entities and relations

        Returns:
            Raw set of entities (str) and relations (tuple of str), and whether
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
        for relation in output_sentence.split(self.HEAD_TOKEN):
            if len(relation) == 0:
                # if '[HEAD]' is at the beginning of the sentence we can obtain an empty str
                continue

            # try splitting head from type and tail
            split_type = relation.split(self.TYPE_TOKEN)
            if len(split_type) != 2:
                format_error = True
                continue
            head, type_and_tail = split_type

            # try splitting type and tail
            split_tail = type_and_tail.split(self.TAIL_TOKEN)
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
