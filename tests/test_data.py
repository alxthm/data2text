from pathlib import Path

from transformers import AutoTokenizer

from src.data.datasets import Seq2seqDataset
from src.data.formatting import Example, Triple, Entity, RelationType, GraphFormat


def test_output_format():
    graph = [
        Triple(
            Entity("Abilene , Texas"),
            RelationType(short="cityServed", natural="city served"),
            Entity("Abilene Regional Airport"),
        )
    ]
    output_format = GraphFormat()
    assert (
        output_format.serialize_graph(graph)
        == "[HEAD] Abilene , Texas [TYPE] city served [TAIL] Abilene Regional Airport"
    )
    graph = [
        Triple(
            Entity("Abilene , Texas"),
            RelationType(short="cityServed", natural="city served"),
            Entity("Abilene Regional Airport"),
        ),
        Triple(
            Entity("Mbappe"),
            RelationType(short="bestPlayer", natural="best player"),
            Entity("France"),
        ),
    ]
    output_format = GraphFormat()
    assert (
        output_format.serialize_graph(graph)
        == "[HEAD] Abilene , Texas [TYPE] city served "
        "[TAIL] Abilene Regional Airport [HEAD] Mbappe [TYPE] best player [TAIL] France"
    )


def test_run_inference():
    output_sentence = (
        "[HEAD] Abilene , Texas [TYPE] city served [TAIL] Abilene Regional Airport"
        "[HEAD] Abilene [TYPE] served [TAIL] Regional Airport"
    )
    output_format = GraphFormat()
    pred_ent, pred_rel, error = output_format.extract_raw_graph(output_sentence)
    assert not error
    assert pred_ent == {
        "Abilene , Texas",
        "Abilene",
        "Abilene Regional Airport",
        "Regional Airport",
    }
    assert pred_rel == {
        ("Abilene , Texas", "city served", "Abilene Regional Airport"),
        ("Abilene", "served", "Regional Airport"),
    }


def test_run_inference2():
    output_sentence = (
        "[HEAD] Abilene , Texas [TYPE] city served [TAIL] Abilene Regional Airport"
        "[HEAD] Abilene served [TAIL] Regional Airport"
    )
    output_format = GraphFormat()
    pred_ent, pred_rel, error = output_format.extract_raw_graph(output_sentence)
    assert error
    assert pred_ent == {"Abilene , Texas", "Abilene Regional Airport"}
    assert pred_rel == {
        ("Abilene , Texas", "city served", "Abilene Regional Airport"),
    }


def test_webnlg_dataset():
    # check that we obtain the desired format when creating WebNLG dataset object
    examples_to_match = [
        Example(
            text="Abilene , Texas is served by the Abilene Regional Airport .",
            graph=[
                Triple(
                    Entity("Abilene Regional Airport"),
                    RelationType(short="cityServed", natural="city served"),
                    Entity("Abilene , Texas"),
                )
            ],
        ),
        Example(
            text="serving size for the Barny Cakes is 30.0 g .",
            graph=[
                Triple(
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
    webnlg = Seq2seqDataset(data_dir=project_dir / "data", split="train", tokenizer=tokenizer)
    examples = webnlg.construct_examples(fake_raw_data)
    assert examples == examples_to_match


if __name__ == "__main__":
    test_output_format()
    test_run_inference()
    test_run_inference2()
    test_webnlg_dataset()
