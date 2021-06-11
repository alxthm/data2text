from typing import List, Tuple

from dataclasses import dataclass


@dataclass
class Entity:
    text: str


@dataclass
class RelationType:
    short: str
    natural: str  # string to use in input/output sequences, more readable


@dataclass
class Relation:
    head: Entity
    type: RelationType
    tail: Entity

    def __repr__(self):
        return f"({self.head.text} -> {self.type.short} -> {self.tail.text})"


@dataclass
class Example:
    text: str  # plain text sentence
    graph: List[Relation]


class InputFormat:
    prefix_t2g = "text to graph : "

    def format_input(self, text: str):
        return self.prefix_t2g + text


class OutputFormat:
    HEAD_TOKEN = "[HEAD]"
    TYPE_TOKEN = "[TYPE]"
    TAIL_TOKEN = "[TAIL]"

    def format_output(self, graph: List[Relation]) -> str:
        """
        Format graph (list of relations) into a string

        Examples
            for a graph with only one relation:
            '[HEAD] Abilene , Texas [TYPE] city served [TAIL] Abilene Regional Airport'

        """
        s = ""
        for r in graph:
            s += " ".join(
                [
                    self.HEAD_TOKEN,
                    r.head.text,
                    self.TYPE_TOKEN,
                    r.type.natural,
                    self.TAIL_TOKEN,
                    r.tail.text,
                ]
            )
        return s

    def run_inference(self, output_sentence: str) -> Tuple[set, set, bool]:
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
            wrong_format:
                False

        """
        wrong_format = False
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
                wrong_format = True
                continue
            head, type_and_tail = split_type

            # try splitting type and tail
            split_tail = type_and_tail.split(self.TAIL_TOKEN)
            if len(split_tail) != 2:
                wrong_format = True
                continue
            type, tail = split_tail

            e1 = head.strip()
            rel = type.strip()
            e2 = tail.strip()
            predicted_entities.update([e1, e2])
            predicted_relations.add((e1, rel, e2))

        return predicted_entities, predicted_relations, wrong_format
