from itertools import islice
from pathlib import Path

from torch.utils.data import DataLoader
from transformers import AutoTokenizer, default_data_collator

from src.data.datasets import WebNLG2020
from src.data.formatting import GraphFormat, add_prefix
from src.data.noise_functions import noise_functions_list
from src.trainer import Seq2seqTrainer

project_dir = Path(__file__).resolve().parents[1]
tokenizer = AutoTokenizer.from_pretrained("t5-base")
dataset = WebNLG2020(data_dir=project_dir / "data", split="dev", tokenizer=tokenizer)
dataloader = DataLoader(
    dataset,
    batch_size=8,
    collate_fn=default_data_collator,
)


def test_text_prefix():
    batch = next(iter(dataloader))
    text_ids = batch["text_ids"]
    text_ids_w_prefix = add_prefix(
        input_ids=text_ids, target="graph", tokenizer=tokenizer, max_seq_len=256
    )
    text = tokenizer.batch_decode(text_ids, skip_special_tokens=True)
    text_w_prefix = tokenizer.batch_decode(text_ids_w_prefix, skip_special_tokens=True)
    for t, t_w_prefix in zip(text, text_w_prefix):
        # should be true, except if t is max_length (and has to be truncated)
        assert t_w_prefix == "Generate graph: " + t


def test_graph_prefix():
    batch = next(iter(dataloader))
    graph_ids = batch["graph_ids"]
    graph_ids_w_prefix = add_prefix(
        input_ids=graph_ids, target="text", tokenizer=tokenizer, max_seq_len=256
    )
    graph = tokenizer.batch_decode(graph_ids, skip_special_tokens=True)
    graph_w_prefix = tokenizer.batch_decode(
        graph_ids_w_prefix, skip_special_tokens=True
    )
    for g, g_w_prefix in zip(graph, graph_w_prefix):
        assert g_w_prefix == "Generate text: " + g


def test_att_mask():
    batch = next(iter(dataloader))
    text_ids = batch["text_ids"]
    graph_ids = batch["graph_ids"]
    att_mask_text = batch["att_mask_text"]
    att_mask_graph = batch["att_mask_graph"]
    assert (att_mask_graph == Seq2seqTrainer.get_att_mask(graph_ids)).all()
    assert (att_mask_text == Seq2seqTrainer.get_att_mask(text_ids)).all()


def test_split_graph():
    seq = (
        "[HEAD] 20 Fenchurch Street [TYPE] location [TAIL] London "
        "[HEAD] London [TYPE] leader title [TAIL] European Parliament"
    )
    assert GraphFormat.split_triples(seq) == [
        "[HEAD] 20 Fenchurch Street [TYPE] location [TAIL] London",
        "[HEAD] London [TYPE] leader title [TAIL] European Parliament",
    ]


def print_noisy_inputs():
    batch = next(islice(iter(dataloader), 150, 151))
    text_ids = batch["text_ids"]
    graph_ids = batch["graph_ids"]
    graphs = tokenizer.batch_decode(graph_ids, skip_special_tokens=True)
    texts = tokenizer.batch_decode(text_ids, skip_special_tokens=True)
    for i in range(len(graphs)):
        print(f"[{i}]")
        graph = graphs[i]
        print("original graph: ", graph)
        for noise_type, f in noise_functions_list.items():
            noisy_seq, _ = f(graph, True)
            print(f"{noise_type}: {noisy_seq}")
        text = texts[i]
        print("original text: ", text)
        for noise_type, f in noise_functions_list.items():
            noisy_seq, _ = f(text, False)
            print(f"{noise_type}: {noisy_seq}")


if __name__ == "__main__":
    test_text_prefix()
    test_graph_prefix()
    test_att_mask()
    test_split_graph()
    print_noisy_inputs()
