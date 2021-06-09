from pathlib import Path

from transformers import AutoTokenizer

from src.data.datasets import WebNLG
from src.data.formatting import Example, Relation, Entity, RelationType, OutputFormat


def test_output_format():
    graph = [
        Relation(
            Entity("Abilene , Texas"),
            RelationType(short="cityServed", natural="city served"),
            Entity("Abilene Regional Airport"),
        )
    ]
    output_format = OutputFormat()
    assert (
        output_format.format_output(graph)
        == "[HEAD] Abilene , Texas [TYPE] city served [TAIL] Abilene Regional Airport"
    )


def test_webnlg_dataset():
    examples_to_match = [
        Example(
            text="Abilene , Texas is served by the Abilene Regional Airport .",
            graph=[
                Relation(
                    Entity("Abilene Regional Airport"),
                    RelationType(short="cityServed", natural="city served"),
                    Entity("Abilene , Texas"),
                )
            ],
        ),
        Example(
            text="serving size for the Barny Cakes is 30.0 g .",
            graph=[
                Relation(
                    Entity("Barny Cakes"),
                    RelationType(short="servingSize", natural="serving size"),
                    Entity("30.0 g"),
                )
            ],
        ),
    ]

    fake_raw_data = [
        {
            "relations": [
                [
                    ["Abilene", "Regional", "Airport"],
                    "cityServed",
                    ["Abilene", ",", "Texas"],
                ]
            ],
            "text": "<ENT_0> is served by the <ENT_1> .",
            "entities": [["Abilene", ",", "Texas"], ["Abilene", "Regional", "Airport"]],
        },
        {
            "relations": [[["Barny", "Cakes"], "servingSize", ["30.0", "g"]]],
            "text": "serving size for the <ENT_0> is <ENT_1> .",
            "entities": [["Barny", "Cakes"], ["30.0", "g"]],
        },
    ]

    project_dir = Path(__file__).resolve().parents[1]
    tokenizer = AutoTokenizer.from_pretrained("t5-small")
    webnlg = WebNLG(
            data_dir=project_dir / "data", split="train", tokenizer=tokenizer
        )
    examples = webnlg.construct_examples(fake_raw_data)
    assert examples == examples_to_match


if __name__ == '__main__':
    test_output_format()
    test_webnlg_dataset()