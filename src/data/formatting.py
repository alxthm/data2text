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
    token_head = "[HEAD]"
    token_type = "[TYPE]"
    token_tail = "[TAIL]"

    def format_output(self, graph: List[Relation]):
        s = ""
        for r in graph:
            s += " ".join(
                [
                    self.token_head,
                    r.head.text,
                    self.token_type,
                    r.type.natural,
                    self.token_tail,
                    r.tail.text,
                ]
            )
        return s
