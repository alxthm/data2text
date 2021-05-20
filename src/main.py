import copy
import datetime
import logging
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import List

import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from omegaconf import OmegaConf
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers.optimization import get_cosine_schedule_with_warmup

from src.data.webnlg import write_txt, prepare_data
from src.model.g2t import GraphWriter
from src.model.t2g import ModelLSTM
from src.utils import WarningsFilter

logging.basicConfig(level=logging.INFO)
logging.info("Start Logging")
bleu = Bleu(4)
meteor = Meteor()
rouge = Rouge()
cider = Cider()
tb_writer: SummaryWriter = None


def fake_sent(x):
    return " ".join(["<ENT_{0:}>".format(xx) for xx in range(len(x))])


# def prep_data(conf, load=""):
#     # prep data always has two steps, build the vocabulary first and then generate data samples
#     train_raw = json.load(open(conf.train_file, "r"))
#     max_len = sorted([len(x["text"].split()) for x in train_raw])[
#         int(0.95 * len(train_raw))
#     ]
#     train_raw = [x for x in train_raw if len(x["text"].split()) < max_len]
#     train_raw = train_raw[: int(len(train_raw) * conf.split)]
#
#     dev_raw = json.load(open(conf.dev_file, "r"))
#     test_raw = json.load(open(conf.test_file, "r"))
#     if len(load) == 0:
#         # scan the data and create vocabulary
#         vocab = scan_data(train_raw)
#         vocab = scan_data(dev_raw, vocab)
#         vocab = scan_data(test_raw, vocab, sp=True)
#         for v in vocab.values():
#             v.build()
#             logging.info(
#                 "Vocab Size {0:}, detached by test set {1:}".format(len(v), len(v.sp))
#             )
#         return vocab
#     else:
#         vocab = torch.load(load)["vocab"]
#
#     logging.info("MAX_LEN {0:}".format(max_len))
#     pool = DataPool()
#     _raw = []
#     for x in train_raw:
#         _x = copy.deepcopy(x)
#         if conf.mode == "sup":
#             _raw.append(_x)
#         else:  # make sure that no information leak in unsupervised settings
#             _x["relations"] = []
#             _raw.append(_x)
#
#     fill_pool(pool, vocab, _raw, "train_g2t")
#     _raw = []
#     for x in train_raw:
#         _x = copy.deepcopy(x)
#         if conf.mode == "sup":
#             _raw.append(_x)
#         else:  # make sure that no information leak in unsupervised settings
#             _x["text"] = fake_sent(_x["entities"])
#             _raw.append(_x)
#
#     fill_pool(pool, vocab, _raw, "train_t2g")
#     _raw = []
#     for x in dev_raw:
#         _x = copy.deepcopy(x)
#         _x["text"] = fake_sent(_x["entities"])
#         _raw.append(_x)
#
#     fill_pool(pool, vocab, dev_raw, "dev")
#     fill_pool(
#         pool, vocab, _raw, "dev_t2g_blind"
#     )  # prepare for the entity2graph setting
#     fill_pool(pool, vocab, test_raw, "test")
#     return pool, vocab


def prep_model(conf, vocab):
    g2t_model = GraphWriter(copy.deepcopy(conf.g2t), vocab)
    t2g_model = ModelLSTM(
        relation_types=len(vocab["relation"]),
        d_model=conf.t2g.nhid,
        dropout=conf.t2g.drop,
    )
    return g2t_model, t2g_model


vae_step = 0.0


def train_g2t_one_step(batch, model, optimizer, conf):
    global vae_step
    model.train()
    optimizer.zero_grad()
    pred, pred_c, kld_loss = model(batch)
    loss = F.nll_loss(
        pred.reshape(-1, pred.shape[-1]), batch["g2t_tgt"].reshape(-1), ignore_index=0
    )
    loss = loss  # + 1.0 * ((1.-pred_c.sum(1))**2).mean() #coverage penalty
    loss = (
        loss
        + min(1.0, (vae_step + 100) / (vae_step + 10000)) * 8.0 * 1.0 / 385 * kld_loss
    )  # magic number
    vae_step += 1
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), conf.clip)
    optimizer.step()
    return loss.item(), kld_loss.item()


def train_t2g_one_step(batch, model, optimizer, conf, device, t2g_weight=None):
    model.train()
    if t2g_weight is not None:
        # category weights
        t2g_weight = torch.from_numpy(t2g_weight).float().to(device)
    optimizer.zero_grad()
    # logits for the relation type, between every entity tuple of the sentence
    pred = model(batch)  # (bs, ne, ne, num_relations)
    loss = F.nll_loss(
        pred.contiguous().view(-1, pred.shape[-1]),
        batch["t2g_tgt"].contiguous().view(-1),  # initially shape (bs, ne, ne)
        ignore_index=0,
        weight=t2g_weight,
    )
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), conf.clip)
    optimizer.step()
    return loss.item()


# def t2g_teach_g2t_one_step(
#     raw_batch, model_t2g, model_g2t, optimizer, conf, vocab, device
# ):
#     # train a g2t model with the synthetic input from t2g model
#     model_t2g.eval()
#     model_g2t.train()
#     batch_t2g = batch2tensor_t2g(raw_batch, device, vocab)
#     with torch.no_grad():
#         t2g_pred = model_t2g(batch_t2g).argmax(-1).cpu()
#     syn_batch = []
#     for _i, sample in enumerate(t2g_pred):
#         rel = []
#         for e1 in range(len(raw_batch[_i]["ent_text"])):
#             for e2 in range(len(raw_batch[_i]["ent_text"])):
#                 try:
#                     if (
#                         sample[e1, e2] != 3 and sample[e1, e2] != 0
#                     ):  # 3 is no relation and 0 is <PAD>
#                         rel.append([e1, int(sample[e1, e2]), e2])
#                 except:
#                     logging.warn(
#                         "{0:}".format(
#                             [
#                                 [vocab["entity"](x) for x in y]
#                                 for y in raw_batch[_i]["ent_text"]
#                             ]
#                         )
#                     )
#                     logging.warn("{0:}".format(sample.size()))
#         _syn = tensor2data_t2g(raw_batch[_i], rel, vocab)
#         syn_batch.append(_syn)
#     if len(syn_batch) == 0:
#         return None
#     batch_g2t = batch2tensor_g2t(syn_batch, device, vocab)
#     loss, kld = train_g2t_one_step(batch_g2t, model_g2t, optimizer, conf.g2t)
#     return loss, kld
#
#
# def g2t_teach_t2g_one_step(
#     raw_batch, model_g2t, model_t2g, optimizer, conf, vocab, device, t2g_weight=None
# ):
#     # train a t2g model with the synthetic input from g2t model
#     model_g2t.eval()
#     model_t2g.train()
#     syn_batch = []
#     if len(raw_batch) > 0:
#         batch_g2t = batch2tensor_g2t(raw_batch, device, vocab)
#         with torch.no_grad():
#             g2t_pred = model_g2t(batch_g2t, beam_size=1).cpu()
#         for _i, sample in enumerate(g2t_pred):
#             _s = sample.tolist()
#             if 2 in _s:  # <EOS> in list
#                 _s = _s[: _s.index(2)]
#             _syn = tensor2data_g2t(raw_batch[_i], _s)
#             syn_batch.append(_syn)
#     batch_t2g = batch2tensor_t2g(syn_batch, device, vocab, add_inp=True)
#     loss = train_t2g_one_step(
#         batch_t2g, model_t2g, optimizer, conf.t2g, device, t2g_weight=t2g_weight
#     )
#     return loss


# def eval_g2t(pool, _type, vocab, model, conf, global_step, device):
#     logging.info("Eval on {0:}".format(_type))
#     model.eval()
#     hyp, ref, _same = [], [], []
#     unq_hyp = {}
#     unq_ref = defaultdict(list)
#     batch_size = 8 * conf.batch_size
#     with tqdm(
#         list(pool.draw_with_type(batch_size, False, _type)),
#     ) as tqb:
#         for i, _batch in enumerate(tqb):
#             with torch.no_grad():
#                 batch = batch2tensor_g2t(_batch, device, vocab)
#                 seq = model(batch, beam_size=conf.beam_size)
#             r = write_txt(batch, batch["tgt"], vocab["text"])
#             h = write_txt(batch, seq, vocab["text"])
#             _same.extend([str(x["raw_relation"]) + str(x["ent_text"]) for x in _batch])
#             hyp.extend(h)
#             ref.extend(r)
#         hyp = [x[0] for x in hyp]
#         ref = [x[0] for x in ref]
#         idxs, _same = list(zip(*sorted(enumerate(_same), key=lambda x: x[1])))
#
#         ptr = 0
#         for i in range(len(hyp)):
#             if i > 0 and _same[i] != _same[i - 1]:
#                 ptr += 1
#             unq_hyp[ptr] = hyp[idxs[i]]
#             unq_ref[ptr].append(ref[idxs[i]])
#
#         max_len = max([len(ref) for ref in unq_ref.values()])
#         unq_hyp = sorted(unq_hyp.items(), key=lambda x: x[0])
#         unq_ref = sorted(unq_ref.items(), key=lambda x: x[0])
#         hyp = [x[1] for x in unq_hyp]
#         ref = [[x.lower() for x in y[1]] for y in unq_ref]
#
#     wf_h = open("hyp.txt", "w", encoding="utf-8")
#     for i, h in enumerate(hyp):
#         wf_h.write(str(h) + "\n")
#     wf_h.close()
#     hyp = dict(zip(range(len(hyp)), [[x.lower()] for x in hyp]))
#     ref = dict(zip(range(len(ref)), ref))
#     ret = bleu.compute_score(ref, hyp)
#
#     metrics = {
#         "BLEU INP": len(hyp),
#         "BLEU 1": ret[0][0],
#         "BLEU 2": ret[0][1],
#         "BLEU 3": ret[0][2],
#         "BLEU 4": ret[0][3],
#         "METEOR": meteor.compute_score(ref, hyp)[0],
#         "ROUGE_L": (rouge.compute_score(ref, hyp)[0]),
#         "Cider": cider.compute_score(ref, hyp)[0],
#     }
#     for k, v in metrics.items():
#         logging.info(f"{k} {v}")
#         tb_writer.add_scalar(k, v, global_step=global_step)
#
#     mlflow.log_artifact("hyp.txt", f"g2t_hyp/{global_step}")
#     mlflow.log_metrics(metrics, step=global_step)
#     return ret[0][-1]
#
#
# def eval_t2g(pool, _type, vocab, model, conf, global_step, device):
#     # evaluate t2g model
#     logging.info("Eval on {0:}".format(_type))
#     hyp, ref, pos_label = [], [], []
#     model.eval()
#     wf = open("t2g_show.txt", "w", encoding="utf-8")
#     with tqdm(
#         list(pool.draw_with_type(conf.batch_size, False, _type)),
#     ) as tqb:
#         for i, _batch in enumerate(tqb):
#             with torch.no_grad():
#                 batch = batch2tensor_t2g(_batch, device, vocab)
#                 pred = model(batch)
#             _pred = pred.view(-1, pred.shape[-1]).argmax(-1).cpu().long().tolist()
#             _gold = batch["tgt"].view(-1).cpu().long().tolist()
#             tpred = pred.argmax(-1).cpu().numpy()
#             tgold = batch["tgt"].cpu().numpy()
#
#             cnts = []
#             for j in range(len(_batch)):
#                 _cnt = 0
#                 ents = [
#                     [y for y in vocab["entity"](x) if y[0] != "<"]
#                     for x in _batch[j]["ent_text"]
#                 ]
#                 wf.write("=====================\n")
#                 rels = []
#                 for e1 in range(len(ents)):
#                     for e2 in range(len(ents)):
#                         if tpred[j, e1, e2] != 3 and tpred[j, e1, e2] != 0:
#                             rels.append((e1, int(tpred[j, e1, e2]), e2))
#                 wf.write(
#                     str(
#                         [
#                             (ents[e1], vocab["relation"](r), ents[e2])
#                             for e1, r, e2 in rels
#                         ]
#                     )
#                     + "\n"
#                 )
#                 rels = []
#                 for e1 in range(len(ents)):
#                     for e2 in range(len(ents)):
#                         if tgold[j, e1, e2] != 3 and tgold[j, e1, e2] != 0:
#                             rels.append((e1, int(tgold[j, e1, e2]), e2))
#                         if tgold[j, e1, e2] > 0:
#                             _cnt += 1
#                 wf.write(
#                     str(
#                         [
#                             (ents[e1], vocab["relation"](r), ents[e2])
#                             for e1, r, e2 in rels
#                         ]
#                     )
#                     + "\n"
#                 )
#                 cnts.append(_cnt)
#
#             pred, gold = [], []
#             for j in range(len(_gold)):
#                 if _gold[j] > 0:  # not the <PAD>
#                     pred.append(_pred[j])
#                     gold.append(_gold[j])
#             pos_label.extend([x for x in gold if x != 3])  # 3 is no relation
#             hyp.extend(pred)
#             ref.extend(gold)
#     wf.close()
#     pos_label = list(set(pos_label))
#
#     f1_micro = f1_score(ref, hyp, average="micro", labels=pos_label, zero_division=0)
#     f1_macro = f1_score(ref, hyp, average="macro", labels=pos_label, zero_division=0)
#
#     logging.info("F1 micro {0:} F1 macro {1:}".format(f1_micro, f1_macro))
#     mlflow.log_artifact("t2g_show.txt", f"t2g_show/{global_step}")
#     mlflow.log_metrics(
#         {f"{_type}_f1_micro": f1_micro, f"{_type}_f1_macro": f1_macro}, step=global_step
#     )
#     tb_writer.add_scalar(f"{_type}_f1_micro", f1_micro, global_step=global_step)
#     tb_writer.add_scalar(f"{_type}_f1_macro", f1_macro, global_step=global_step)
#     return f1_micro
#


def supervise(
    batch,
    model_g2t,
    model_t2g,
    optimizerG2T,
    optimizerT2G,
    conf,
    t2g_weight,
    vocab,
    device,
):
    model_g2t.blind, model_t2g.blind = False, False
    # batch = batch2tensor_t2g(batch_t2g, device, vocab)
    _loss1 = train_t2g_one_step(
        batch, model_t2g, optimizerT2G, conf.t2g, device, t2g_weight=t2g_weight
    )
    # batch = batch2tensor_g2t(batch_g2t, device, vocab)
    _loss2, kld = train_g2t_one_step(batch, model_g2t, optimizerG2T, conf.g2t)
    return _loss1, _loss2, kld


def write_t2g_log(wf, rel_vocab, rel_pred: np.array, ents: List[List]):
    """
    Write a predicted graph as plain text to the logging file

    Args:
        rel_pred: numpy array of shape (max_num_ent, max_num_ent).
            Contains the indices of predicted relations between entities.
        ents: list of entities for this example, each entity being a list of tokens

    """
    rels = []
    for e1 in range(len(ents)):
        for e2 in range(len(ents)):
            # 0: padding, 3: unknown
            if rel_pred[e1, e2] != 3 and rel_pred[e1, e2] != 0:
                rels.append((e1, int(rel_pred[e1, e2]), e2))
    wf.write(str([(ents[e1], rel_vocab(r), ents[e2]) for e1, r, e2 in rels]) + "\n")


def evaluate(dataloader_val, t2g_model, g2t_model, vocab, global_step, beam_size):
    ent_vocab = vocab["entity"]
    rel_vocab = vocab["relation"]
    text_vocab = vocab["text"]
    wf_t2g = open("/tmp/t2g_out.txt", "w", encoding="utf-8")
    t2g_hyp, t2g_ref, t2g_pos_label = [], [], []
    g2t_hyp = []
    g2t_ref = []
    g2t_same = []
    # todo: use pytorch metrics?

    for i, batch in tqdm(enumerate(dataloader_val)):
        batch_size = len(batch["original_ent_text"])

        # t2g
        pred = t2g_model(batch)
        _pred = pred.view(-1, pred.shape[-1]).argmax(-1).cpu().long().tolist()
        _gold = batch["t2g_tgt"].view(-1).cpu().long().tolist()
        tpred = pred.argmax(-1).cpu().numpy()
        tgold = batch["t2g_tgt"].cpu().numpy()
        # save predictions as plain text
        for j in range(batch_size):
            ents = [
                [y for y in ent_vocab(x) if y[0] != "<"]
                for x in batch["original_ent_text"][j]
            ]
            wf_t2g.write("=====================\n")
            wf_t2g.write("--- Predictions\n")
            write_t2g_log(wf_t2g, rel_vocab, tpred[j], ents)
            wf_t2g.write("--- Target\n")
            write_t2g_log(wf_t2g, rel_vocab, tgold[j], ents)
        # compute f1 metrics
        pred, gold = [], []
        for j in range(len(_gold)):
            if (
                _gold[j] > 0
            ):  # ignore <PAD> -> todo: simply use ignore_index of F1 metric?
                pred.append(_pred[j])
                gold.append(_gold[j])
        t2g_pos_label.extend([x for x in gold if x != 3])  # 3 is no relation
        t2g_hyp.extend(pred)
        t2g_ref.extend(gold)

        # g2t
        seq = g2t_model(batch, beam_size=beam_size)  # (bs, max_sent_len)
        r = write_txt(batch, batch["g2t_tgt"], text_vocab)
        h = write_txt(batch, seq, text_vocab)
        g2t_same.extend(
            [
                str(batch["original_raw_relation"][i])
                + str(batch["original_ent_text"][i])
                for i in range(batch_size)
            ]
        )
        # save text predictions for evaluation and logging at the end of val epoch
        g2t_hyp.extend(h)
        g2t_ref.extend(r)

    # compute metrics over all validation batch and log results
    wf_t2g.close()
    pos_label = list(set(t2g_pos_label))
    f1_micro = f1_score(
        t2g_ref, t2g_hyp, average="micro", labels=pos_label, zero_division=0
    )
    f1_macro = f1_score(
        t2g_ref, t2g_hyp, average="macro", labels=pos_label, zero_division=0
    )
    mlflow.log_artifact("/tmp/t2g_out.txt", f"t2g_out/{global_step}")
    mlflow.log_metrics(
        {"dev_f1_micro": f1_micro, "dev_f1_macro": f1_macro}, step=global_step
    )
    tb_writer.add_scalar("dev_f1_micro", f1_micro, global_step=global_step)
    tb_writer.add_scalar("dev_f1_macro", f1_macro, global_step=global_step)

    # g2t - format predictions into text
    unq_hyp = {}
    unq_ref = defaultdict(list)
    g2t_hyp = [x[0] for x in g2t_hyp]
    g2t_ref = [x[0] for x in g2t_ref]
    idxs, g2t_same = list(zip(*sorted(enumerate(g2t_same), key=lambda x: x[1])))
    ptr = 0
    for i in range(len(g2t_hyp)):
        if i > 0 and g2t_same[i] != g2t_same[i - 1]:
            ptr += 1
        unq_hyp[ptr] = g2t_hyp[idxs[i]]
        unq_ref[ptr].append(g2t_ref[idxs[i]])
    unq_hyp = sorted(unq_hyp.items(), key=lambda x: x[0])
    unq_ref = sorted(unq_ref.items(), key=lambda x: x[0])
    g2t_hyp = [x[1] for x in unq_hyp]
    g2t_ref = [[x.lower() for x in y[1]] for y in unq_ref]

    # log predictions
    assert len(unq_ref) == len(unq_hyp)
    with open("/tmp/g2t_out.txt", "w", encoding="utf-8") as wf_g2t:
        for i in range(len(g2t_hyp)):
            wf_g2t.write(
                f"{i}\n" f"pred: {str(g2t_hyp[i])}\n" f"tgt: {str(g2t_ref[i])}\n"
            )
    mlflow.log_artifact("/tmp/g2t_out.txt", f"g2t_out/{global_step}")
    # compute NLG metrics
    g2t_hyp = dict(zip(range(len(g2t_hyp)), [[x.lower()] for x in g2t_hyp]))
    g2t_ref = dict(zip(range(len(g2t_ref)), g2t_ref))
    bleu_score = bleu.compute_score(g2t_ref, g2t_hyp)
    metrics = {
        "BLEU INP": len(g2t_hyp),
        "BLEU 1": bleu_score[0][0],
        "BLEU 2": bleu_score[0][1],
        "BLEU 3": bleu_score[0][2],
        "BLEU 4": bleu_score[0][3],
        "METEOR": meteor.compute_score(g2t_ref, g2t_hyp)[0],
        "ROUGE_L": (rouge.compute_score(g2t_ref, g2t_hyp)[0]),
        "Cider": cider.compute_score(g2t_ref, g2t_hyp)[0],
    }
    mlflow.log_metrics(metrics)
    for k, v in metrics.items():
        # logging.info(f"{k} {v}")
        tb_writer.add_scalar(k, v, global_step=global_step)


# def back_translation(
#     batch_g2t,
#     batch_t2g,
#     model_g2t,
#     model_t2g,
#     optimizerG2T,
#     optimizerT2G,
#     conf,
#     t2g_weight,
#     vocab,
#     device,
# ):
#     model_g2t.blind, model_t2g.blind = False, False
#     _loss1 = g2t_teach_t2g_one_step(
#         batch_t2g,
#         model_g2t,
#         model_t2g,
#         optimizerT2G,
#         conf,
#         vocab,
#         device,
#         t2g_weight=t2g_weight,
#     )
#     _loss2, kld = t2g_teach_g2t_one_step(
#         batch_g2t, model_t2g, model_g2t, optimizerG2T, conf, vocab, device
#     )
#     return _loss1, _loss2, kld
#


def train(conf, device, dataloader_train, dataloader_val, dataloader_test, vocab):
    print("Preparing model...")
    model_g2t, model_t2g = prep_model(conf, vocab)
    model_g2t.to(device)
    model_t2g.to(device)

    optimizerG2T = torch.optim.Adam(
        model_g2t.parameters(),
        lr=conf.g2t.lr,
        weight_decay=conf.g2t.weight_decay,
    )
    schedulerG2T = get_cosine_schedule_with_warmup(
        optimizer=optimizerG2T,
        num_warmup_steps=400,
        num_training_steps=800 * conf.main.epoch,
    )
    optimizerT2G = torch.optim.Adam(
        model_t2g.parameters(),
        lr=conf.t2g.lr,
        weight_decay=conf.t2g.weight_decay,
    )
    schedulerT2G = get_cosine_schedule_with_warmup(
        optimizer=optimizerT2G,
        num_warmup_steps=400,
        num_training_steps=800 * conf.main.epoch,
    )
    loss_t2g, loss_g2t = [], []
    klds = []

    t2g_weight = [vocab["relation"].wf.get(x, 0) for x in vocab["relation"].i2s]
    t2g_weight[0] = 0
    max_w = max(t2g_weight)
    t2g_weight = np.array(t2g_weight).astype("float32")
    t2g_weight = (max_w + 1000) / (t2g_weight + 1000)

    global_step = 0
    for i in range(0, conf.main.epoch):
        model_t2g.train()
        model_g2t.train()

        # _data_g2t = list(pool.draw_with_type(conf.main.batch_size, True, "train_g2t"))
        # _data_t2g = list(pool.draw_with_type(conf.main.batch_size, True, "train_t2g"))
        # data_list = list(zip(_data_g2t, _data_t2g))
        # _data = data_list
        # with tqdm(_data) as tqb:
        #     for j, batch in enumerate(tqb):
        for j, batch in enumerate(tqdm(dataloader_train)):
            _loss1, _loss2, kld = supervise(
                batch,
                model_g2t,
                model_t2g,
                optimizerG2T,
                optimizerT2G,
                conf,
                t2g_weight,
                vocab,
                device,
            )
            loss_t2g.append(_loss1)
            schedulerT2G.step()
            loss_g2t.append(_loss2)
            schedulerG2T.step()
            klds.append(kld)
            metrics = {
                "train_loss_t2g": np.mean(loss_t2g),
                "train_loss_g2t": np.mean(loss_g2t),
                "train_kld": np.mean(klds),
            }
            mlflow.log_metrics(metrics, step=global_step)
            for k, v in metrics.items():
                tb_writer.add_scalar(k, v, global_step=global_step)
            # tqb.set_postfix(metrics)
            global_step += 1

        logging.info("Epoch " + str(i))

        # validation step
        model_g2t.blind, model_t2g.blind = False, False
        model_t2g.eval()
        model_g2t.eval()
        with torch.no_grad():
            evaluate(
                dataloader_val,
                model_t2g,
                model_g2t,
                vocab,
                global_step,
                conf.g2t.beam_size,
            )
            if i % 15 == 0:
                # save model checkpoints
                t2g_save = f"{conf.t2g.save}_ep{i}"
                torch.save(model_t2g.state_dict(), t2g_save)
                mlflow.log_artifact(t2g_save, t2g_save)
                g2t_save = f"{conf.g2t.save}_ep{i}"
                torch.save(model_g2t.state_dict(), g2t_save)
                mlflow.log_artifact(g2t_save, g2t_save)

    # test
    torch.save(model_g2t.state_dict(), f"{conf.g2t.save}_final")
    torch.save(model_t2g.state_dict(), f"{conf.t2g.save}_final")
    evaluate(
        dataloader_test,
        model_t2g,
        model_g2t,
        vocab,
        global_step,
        conf.g2t.beam_size,
    )


def main(timestamp: str):
    device = torch.device(0) if torch.cuda.is_available() else torch.device("cpu")

    # Load config
    project_dir = Path(__file__).resolve().parents[1]
    conf = OmegaConf.load(project_dir / "conf/config.yaml")
    print(OmegaConf.to_yaml(conf))

    # seed everything
    random.seed(conf.main.seed)
    torch.manual_seed(conf.main.seed)
    np.random.seed(conf.main.seed)
    torch.cuda.manual_seed_all(conf.main.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # if not os.path.isfile("tmp_vocab.pt"):
    #     vocab = prep_data(conf.main)
    #     torch.save({"vocab": vocab}, "tmp_vocab.pt")

    mlflow.set_tracking_uri("https://mlflow.par.prod.crto.in/")
    # mlflow.create_experiment(
    #     "al.thomas_data_2_text", "hdfs://prod-am6/user/al.thomas/mlflow_artifacts"
    # )
    mlflow.set_experiment("al.thomas_data_2_text")
    # todo: batch updates to speed up training
    #  https://www.mlflow.org/docs/latest/python_api/mlflow.tracking.html#mlflow.tracking.MlflowClient.log_batch
    run_name = f"{timestamp}-{conf.main.mode}"
    global tb_writer
    tb_writer = SummaryWriter(log_dir=str(project_dir / f"models/{run_name}"))
    print(f"run_name: {run_name}")

    with mlflow.start_run(run_name=run_name):
        for k1 in conf.keys():
            for k2 in conf[k1].keys():
                mlflow.log_param(f"{k1}_{k2}", conf[k1][k2])
        # save source files and vocab
        for f in (project_dir / "src").rglob("*.py"):
            mlflow.log_artifact(str(f), f"code/{f}")
        mlflow.log_artifact(str(project_dir / "tmp_vocab.pt"), f"code/tmp_vocab.pt")

        # load data
        dataset_train, dataset_val, dataset_test, vocab, collate_fn = prepare_data(
            data_dir=project_dir / "data", device=device
        )
        batch_size = conf.g2t.batch_size
        assert batch_size == conf.t2g.batch_size
        dataloader_train = DataLoader(
            dataset_train,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )
        dataloader_val = DataLoader(
            dataset_val,
            batch_size=batch_size,
            collate_fn=collate_fn,
        )
        dataloader_test = DataLoader(
            dataset_test,
            batch_size=batch_size,
            collate_fn=collate_fn,
        )

        # train
        train(
            conf=conf,
            device=device,
            dataloader_train=dataloader_train,
            dataloader_val=dataloader_val,
            dataloader_test=dataloader_test,
            vocab=vocab,
        )


if __name__ == "__main__":
    # filter out some hdfs warnings (only those from python libraries)
    sys.stdout = WarningsFilter(sys.stdout)
    sys.stderr = WarningsFilter(sys.stderr)
    timestamp = datetime.datetime.today().strftime("%m%d%H%M%S")
    main(timestamp=timestamp)
