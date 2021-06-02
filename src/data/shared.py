from pathlib import Path
from typing import List

import copy

import dgl
import torch
import uuid

from torch.utils.data import Dataset
from tqdm import tqdm

NODE_TYPE = {"entity": 0, "root": 1, "relation": 2}


class Example(object):
    def __init__(self, data, vocab):
        self.uuid = uuid.uuid4()
        self.vocab = vocab
        self.text = [vocab["text"](x) for x in data["text"].split()]
        self.entities = [vocab["entity"](x) for x in data["entities"]]
        self.relations = []
        for r in data["relations"]:
            e1, e2 = vocab["entity"](r[0]), vocab["entity"](r[2])
            rel = vocab["relation"](r[1])
            e1, e2 = self.entities.index(e1), self.entities.index(e2)
            self.relations.append([e1, rel, e2])

        self.graph = None
        self.graph = build_graph(len(self.entities), self.relations)
        self.id = None

    def __str__(self):
        return "\n".join([str(k) + ":\t" + str(v) for k, v in self.__dict__.items()])

    def __len__(self):
        return len(self.text)

    def get(self):
        if hasattr(self, "_cached_tensor") and False:
            return self._cached_tensor
        else:
            vocab = self.vocab
            ret = {}
            ret["text"] = (
                [vocab["text"]("<BOS>")] + self.text + [vocab["text"]("<EOS>")]
            )
            ret["ent_text"] = [
                [vocab["entity"]("<BOS>")] + x + [vocab["entity"]("<EOS>")]
                for x in self.entities
            ]
            ret["relation"] = [vocab["relation"]("<ROOT>")] + sum(
                [[x[1], vocab["relation"].get_inv(x[1])] for x in self.relations], []
            )
            ret["raw_relation"] = self.relations
            ret["graph"] = self.graph
            ret["uuid"] = self.uuid

            self._cached_tensor = ret
            return self._cached_tensor


class Vocab(object):
    def __init__(
        self,
        max_vocab=2 ** 31,
        min_freq=-1,
        sp=["<PAD>", "<BOS>", "<EOS>", "<UNK>", "<ROOT>"],
    ):
        self.i2s = []
        self.s2i = {}
        self.wf = {}
        self.inv = {}
        self.max_vocab, self.min_freq, self.sp = max_vocab, min_freq, copy.deepcopy(sp)

    def __len__(self):
        return len(self.i2s)

    def __str__(self):
        return "Total " + str(len(self.i2s)) + str(self.i2s[:10])

    def merge(self, _vocab):
        self.wf.update(_vocab.wf)
        self.inv.update(_vocab.inv)
        self.sp = list(set(self.sp + _vocab.sp))

    def update(self, token, inv=False, is_test=False):
        if isinstance(token, list):
            for t in token:
                self.update(t, inv=inv, is_test=is_test)
        else:
            self.wf[token] = self.wf.get(token, 0) + 1
            if inv:
                self.wf[token + "_INV"] = self.wf.get(token + "_INV", 0) + 1
                self.inv[token] = token + "_INV"
            if is_test and token not in self.sp:
                self.sp.append(token)

    def get_inv(self, idx):
        return self.__call__(self.inv.get(self.i2s[idx], "<UNK>"))

    def build(self):
        self.i2s.extend(self.sp)
        sort_kv = sorted(self.wf.items(), key=lambda x: x[1], reverse=True)
        for k, v in sort_kv:
            if (
                len(self.i2s) < self.max_vocab
                and v >= self.min_freq
                and k not in self.sp
            ):
                self.i2s.append(k)
        self.s2i.update(list(zip(self.i2s, range(len(self.i2s)))))

    def __call__(self, x, ents=[]):
        if isinstance(x, list):
            return [self(y) for y in x]
        if isinstance(x, int):
            if x >= len(self.i2s):
                return ents[int(x - len(self.i2s))]
            return self.i2s[x]
        else:
            if x[0] == "<" and x[-1] == ">" and "_" in x:
                try:
                    t = len(self.s2i) + int(x.split("_")[1][:-1])
                except:
                    print(x)
                return len(self.s2i) + int(x.split("_")[1][:-1])
            return self.s2i.get(x, self.s2i["<UNK>"])


class CycleCVAECollator:
    def __init__(
        self, ent_vocab: Vocab, text_vocab: Vocab, tokenizer, device, is_eval: bool
    ):
        self.ent_vocab = ent_vocab
        self.text_vocab = text_vocab
        self.tokenizer = tokenizer
        self.device = device
        self.is_eval = is_eval

    def __call__(self, batch: List):
        if self.is_eval:
            # original raw batch (needed for logging)
            batch_raw = {
                "ent_text": [x["ent_text"] for x in batch],
                "raw_relation": [x["raw_relation"] for x in batch],
            }
            batch_t2g = get_t2g_batch(
                batch, self.ent_vocab, self.text_vocab, self.device, self.tokenizer
            )
            batch_g2t = get_g2t_batch(batch, self.ent_vocab, self.device)
            return batch_raw, batch_t2g, batch_g2t
        else:
            return batch


class CycleCVAEDataset(Dataset):
    def __init__(self, file_path: Path, mode: str):
        """

        Args:
            file_path: Path to the list of examples
            mode: Can be 'text_only', 'graph_only' or 'both'.
                If graph_only, the text will be removed to ensure to information leak
                If text_only, the graph will me removed
        """
        self.data: List[dict] = torch.load(file_path)
        if mode == "graph_only":
            for i in range(len(self.data)):
                self.data[i]["text"] = []
        elif mode == "text_only":
            for i in range(len(self.data)):
                # todo: ok to remove 'graph' as well?
                self.data[i]["relations"] = []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def get_t2g_batch(batch: List, ent_vocab: Vocab, text_vocab: Vocab, device, tokenizer):
    """
    Takes (x,y) from a list of data examples and return a batch of transformed (x,y) where
        - x: 'sents' and 'ents' are computed using 'text', 'ent_text'
        - y: 'tgt' is computed using this and 'raw_relation'

    Args:
        batch: List of data examples
        ent_vocab:
        text_vocab:
        device:

    Returns:

    """
    ent_pos = []
    text = []
    tgt = []
    MAX_ENT = 100
    ent_len = 1
    for data in batch:
        ents = [ent_vocab(x) for x in data["ent_text"]]
        st, ed = [], []
        cstr = ""
        ent_order = []
        for i, t in enumerate(
            data["text"]
        ):  # t: token id of each word in the data["text"] sentence
            if t >= len(text_vocab):
                # if t is the id of an entity token ("ENT_0", ...):
                #   - add the entity text to cstr (without '<BOS>', '<EOS>' tokens)
                #   - add to st and ed the indices of start and end characters
                #       (in sentence cstr), for every entity
                ff = (
                    t - len(text_vocab)
                ) not in ent_order  # t - len(text_vocab) = entity number
                if ff:
                    st.append(len(cstr))
                cstr += " ".join(
                    [x for x in text_vocab(t, ents) if x[0] != "<" and x[-1] != ">"]
                )
                if ff:
                    ent_order.append(
                        t - len(text_vocab)
                    )  # register entities in the order they appear
                    ed.append(len(cstr))
            else:
                if text_vocab(t)[0] == "<":
                    continue
                cstr += text_vocab(t)
            cstr += "" if i == len(data["text"]) - 1 else " "
        # never used?
        # if add_inp:
        #     cstr += " " + " ".join([" ".join(e) for e in ents])
        tok_abs = ["[CLS]"] + tokenizer.tokenize(cstr) + ["[SEP]"]
        _ent_pos = []
        for s, e in zip(st, ed):
            # from start/end indices of entities (s,e) in cstr, find new_s, new_e
            # the start/end indices of entities in tok_abs, the tokenized version of cstr
            guess_start = s - cstr[:s].count(" ") + 5  # ??
            guess_end = e - cstr[:e].count(" ") + 5

            new_s = -1
            new_e = -1
            l = 0
            r = 0
            for i in range(len(tok_abs)):
                # l, r: start/end indices of the text representation of tok_abs[i] in cstr
                l = r
                r = l + len(tok_abs[i]) - tok_abs[i].count("##") * 2
                if l <= guess_start and guess_start < r:
                    new_s = i
                if l <= guess_end and guess_end < r:
                    new_e = i
            _ent_pos.append((new_s, new_e))
        # order the list _ent_pos (start/end indices of every entity in tok_abs)
        # so they are in the order of entity number (ENT_0, ENT_1, etc)
        # instead of the order they appear in the sentence
        _order_ent_pos = []
        for _e in range(len(ents)):
            if (
                _e in ent_order
            ):  # ent_order = [1, 0, 2] if text = "ENT_1 blabla ENT_0 blabla ENT_2"
                idx = ent_order.index(_e)
                _order_ent_pos.append(_ent_pos[idx])
            else:
                idx = 0
                _order_ent_pos.append((0, 1))

        ent_pos.append(_order_ent_pos)
        text.append(tokenizer.convert_tokens_to_ids(tok_abs))
        _tgt = torch.zeros(
            MAX_ENT, MAX_ENT
        )  # 0: <PAD> (in all vocabs, including vocab["relation"])
        _tgt[: len(_ent_pos), : len(_ent_pos)] += 3  # 3: <UNK>
        for _e1, _r, _e2 in data["raw_relation"]:
            if (
                _e1 not in ent_order or _e2 not in ent_order
            ):  # the synthetic data may lose some entities
                continue
            _tgt[_e1, _e2] = _r
        tgt.append(_tgt)
        ent_len = max(ent_len, len(_order_ent_pos))
    batch_t2g = {
        # indices of bert tokens for the full sentences (entities as plain text), padded
        "sents": pad([torch.LongTensor(x) for x in text], "tensor").to(device),
        # for every sentence of the batch, list of start/end indices for ENT_0, ENT_1, etc
        # (in tokenized sentence)
        "ents": ent_pos,
        # shape (batch_size, max_ent_len, max_ent_len) - indexes of relations between
        # each entities i, j
        # (0 if there is no ENT_i or ENT_j in the data sample, 3 if there is no relation
        # between ENT_i and ENT_j)
        "tgt": torch.stack(tgt, 0)[:, :ent_len, :ent_len].long().to(device),
    }
    return batch_t2g


def get_g2t_batch(raw_batch: List, ent_vocab: Vocab, device):
    """
    Take (x,y) from a list of data examples and return a batch of transformed (x,y), where
        - y: 'ent_len', 'raw_ent_len', 'ent_text', 'rel', 'graph' are computed
        using 'ent_text', 'relation' and 'graph'
        - x: 'tgt' and 'text' are computed using 'text'

    Args:
        raw_batch: List of data examples.
        ent_vocab:
        device:

    Returns:

    """
    # list of length bs, with list of sentence entities tokenized
    # (bs, num_ent_i, ent_len_ij)
    ents = [ent_vocab(x["ent_text"]) for x in raw_batch]
    # size (bs, max_num_rel)
    rel = pad([torch.LongTensor(x["relation"]) for x in raw_batch], "tensor").to(device)
    # flattened list of entity token indices tensors
    # (size sum(num_ent_i), for every batch element and for every
    #  entity in the sentence)
    ent_text = sum(
        [[torch.LongTensor(y) for y in x["ent_text"]] for x in raw_batch], []
    )
    # (bs, max_sentence_len)
    text = pad([torch.LongTensor(x["text"]) for x in raw_batch], "tensor").to(device)
    batch_g2t = {
        # -- y
        "ent_len": [len(x["ent_text"]) for x in raw_batch],
        "raw_ent_text": ents,
        # size (sum(num_ent_i), max_entity_len)
        "ent_text": pad(ent_text, "tensor").to(device),
        "rel": rel,
        "graph": dgl.batch([x["graph"] for x in raw_batch]).to(device),
        # -- x
        "tgt": text[:, 1:],  # (bs, max_sentence_len - 1)
        "text": text[:, :-1],
    }
    return batch_g2t


def pad(var_len_list, out_type="list", flatten=False):
    # padding sequences
    if flatten:
        lens = [len(x) for x in var_len_list]
        var_len_list = sum(var_len_list, [])
    max_len = max([len(x) for x in var_len_list])
    if out_type == "list":
        if flatten:
            return [x + ["<PAD>"] * (max_len - len(x)) for x in var_len_list], lens
        else:
            return [x + ["<PAD>"] * (max_len - len(x)) for x in var_len_list]
    if out_type == "tensor":
        if flatten:
            return (
                torch.stack(
                    [
                        torch.cat(
                            [
                                x,
                                torch.zeros(
                                    [max_len - len(x)] + list(x.shape[1:])
                                ).type_as(x),
                            ],
                            0,
                        )
                        for x in var_len_list
                    ],
                    0,
                ),
                lens,
            )
        else:
            return torch.stack(
                [
                    torch.cat(
                        [
                            x,
                            torch.zeros([max_len - len(x)] + list(x.shape[1:])).type_as(
                                x
                            ),
                        ],
                        0,
                    )
                    for x in var_len_list
                ],
                0,
            )


def write_txt(raw_ent_text, seqs, text_vocab):
    """

    Convert the prediction to real text.

    Args:
        raw_ent_text:
        seqs: (bs, max_sent_len) tensor of predicted tokens (words+entities)
        text_vocab:

    Returns:
        List of predictions as plain text, shape: (bs, 1)

    """

    ret = []
    for b, seq in enumerate(seqs):  # b: index of seq in the batch
        txt = []

        for token in seq:
            # copy the entity
            if token >= len(text_vocab):
                if (token - len(text_vocab)) >= len(raw_ent_text[b]):
                    print((token - len(text_vocab)), len(raw_ent_text[b]))
                    tok = ["NO_ENT"]
                else:
                    tok = raw_ent_text[b][token - len(text_vocab)]
                    # tok = ['ENT_'+str(int(token-len(text_vocab)))+'_ENT']
                ent_text = tok
                ent_text = filter(lambda x: x != "<PAD>", ent_text)
                txt.extend(ent_text)
            else:
                if int(token) not in [
                    text_vocab(x) for x in ["<PAD>", "<BOS>", "<EOS>"]
                ]:
                    txt.append(text_vocab(int(token)))
            if int(token) == text_vocab("<EOS>"):
                break
        ret.append(
            [" ".join([str(x) for x in txt]).replace("<BOS>", "").replace("<EOS>", "")]
        )
    return ret


def scan_data(datas, vocab=None, is_test=False):
    MF = -1

    if vocab is None:
        vocab = {
            "text": Vocab(min_freq=MF),
            "entity": Vocab(min_freq=MF),
            "relation": Vocab(),
        }
    for data in tqdm(datas):
        vocab["text"].update(data["text"].split(), is_test=is_test)
        vocab["entity"].update(sum(data["entities"], []), is_test=is_test)
        vocab["relation"].update([x[1] for x in data["relations"]], inv=True)
    return vocab


def get_graph(ent_len, rel_len, adj_edges):
    graph = dgl.DGLGraph()

    # add one node for each entity
    graph.add_nodes(ent_len, {"type": torch.ones(ent_len) * NODE_TYPE["entity"]})
    # add 1 root node
    graph.add_nodes(1, {"type": torch.ones(1) * NODE_TYPE["root"]})
    # add one node for each relation
    graph.add_nodes(
        rel_len * 2, {"type": torch.ones(rel_len * 2) * NODE_TYPE["relation"]}
    )

    # add a bidirectional relation between the root node and all entities
    graph.add_edges(ent_len, torch.arange(ent_len))
    graph.add_edges(torch.arange(ent_len), ent_len)
    # add a relation for each node to itself
    graph.add_edges(
        torch.arange(ent_len + 1 + rel_len * 2), torch.arange(ent_len + 1 + rel_len * 2)
    )
    if len(adj_edges) > 0:
        # for each relation in adj_edges [i, j] (between entity i and j), add relation to the graph
        graph.add_edges(*list(map(list, zip(*adj_edges))))
    return graph


def build_graph(ent_len, relations):
    rel_len = len(relations)

    adj_edges = []
    for i, r in enumerate(relations):
        # for each entities A,B with relation r, add relations
        # [(A,u), (u,B), (B,v), (v,A)]
        # with u, v the two graph nodes that represent relation r
        st_ent, rt, ed_ent = r
        # according to the edge_softmax operator, we need to reverse the graph
        adj_edges.append([ent_len + 1 + 2 * i, st_ent])
        adj_edges.append([ed_ent, ent_len + 1 + 2 * i])
        adj_edges.append([ent_len + 1 + 2 * i + 1, ed_ent])
        adj_edges.append([st_ent, ent_len + 1 + 2 * i + 1])

    graph = get_graph(ent_len, rel_len, adj_edges)
    return graph
