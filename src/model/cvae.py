from collections import defaultdict
from typing import List

import numpy as np
import torch
import torchmetrics
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pytorch_lightning import LightningModule
from torch import nn
from transformers import get_cosine_schedule_with_warmup
import torch.nn.functional as F
from src.data.webnlg import Vocab, batch2tensor_t2g, batch2tensor_g2t, write_txt
from src.model.g2t import G2T
from src.model.t2g import T2G


class CycleCVAE(LightningModule):
    def __init__(
        self,
        text_vocab: Vocab,
        ent_vocab: Vocab,
        rel_vocab: Vocab,
        tokenizer,
        tot_epoch: int,
        dim_h: int,
        dim_z: int,
        enc_lstm_layers: int,
        n_head: int,
        head_dim: int,
        emb_dropout: float,
        attn_drop: float,
        drop: float,
        n_layers_gat: int,
        g2t_lr: float,
        g2t_weight_decay: float,
        beam_size: int,
        t2g_drop: float,
        t2g_lr: float,
        t2g_weight_decay: float,
        gradient_clip_val: float,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.tot_epoch = tot_epoch
        self.text_vocab = text_vocab
        self.ent_vocab = ent_vocab
        self.rel_vocab = rel_vocab
        self.t2g_lr = t2g_lr
        self.t2g_weight_decay = t2g_weight_decay
        self.g2t_lr = g2t_lr
        self.g2t_weight_decay = g2t_weight_decay
        self.beam_size = beam_size
        self.gradient_clip_val = gradient_clip_val

        # category weights: give less importance to more frequent relations
        t2g_weight = [rel_vocab.wf.get(x, 0) for x in rel_vocab.i2s]
        t2g_weight[0] = 0
        max_w = max(t2g_weight)
        t2g_weight = np.array(t2g_weight).astype("float32")
        t2g_weight = (max_w + 1000) / (t2g_weight + 1000)
        self.t2g_weight = torch.from_numpy(t2g_weight).float().to(self.device)

        # todo: check this is the right number of classes
        self.f1_micro = torchmetrics.F1(num_classes=len(rel_vocab), average="micro")
        self.f1_macro = torchmetrics.F1(num_classes=len(rel_vocab), average="macro")

        self.g2t_model = G2T(
            text_vocab=text_vocab,
            ent_vocab=ent_vocab,
            rel_vocab=rel_vocab,
            dim_h=dim_h,
            dim_z=dim_z,
            enc_lstm_layers=enc_lstm_layers,
            n_head=n_head,
            head_dim=head_dim,
            emb_dropout=emb_dropout,
            attn_drop=attn_drop,
            drop=drop,
            n_layers_gat=n_layers_gat,
        )
        self.t2g_model = T2G(
            relation_types=len(rel_vocab), d_model=dim_h, dropout=t2g_drop
        )

    def configure_optimizers(self):
        g2t_optimizer = torch.optim.Adam(
            self.g2t_model.parameters(),
            lr=self.g2t_lr,
            weight_decay=self.g2t_weight_decay,
        )
        t2g_optimizer = torch.optim.Adam(
            self.t2g_model.parameters(),
            lr=self.t2g_lr,
            weight_decay=self.t2g_weight_decay,
        )

        # learning rate schedulers
        g2t_scheduler = get_cosine_schedule_with_warmup(
            optimizer=g2t_optimizer,
            num_warmup_steps=400,
            num_training_steps=800 * self.tot_epoch,
        )
        t2g_scheduler = get_cosine_schedule_with_warmup(
            optimizer=t2g_optimizer,
            num_warmup_steps=400,
            num_training_steps=800 * self.tot_epoch,
        )

        return (t2g_optimizer, g2t_optimizer), (t2g_scheduler, g2t_scheduler)

    def forward(self, *args, **kwargs):
        pass

    def training_step(self, batch, batch_idx):
        (t2g_opt, g2t_opt), (t2g_sch, g2t_sch) = self.configure_optimizers()

        # --- SUPERVISED
        self.g2t_model.blind, self.t2g_model.blind = False, False  # not necessary
        batch_t2g = batch2tensor_t2g(
            batch, self.device, self.text_vocab, self.ent_vocab, self.tokenizer
        )
        batch_g2t = batch2tensor_g2t(batch, self.device, self.ent_vocab)

        # --- train t2g one step
        pred = self.t2g_model(batch_t2g)  # pred (bs, ne, ne, num_relations)
        loss_t2g = F.nll_loss(
            pred.contiguous().view(-1, pred.shape[-1]),
            batch_t2g["tgt"].contiguous().view(-1),  # initially shape (bs, ne, ne)
            ignore_index=0,
            weight=self.t2g_weight,
        )
        t2g_opt.zero_grad()
        self.manual_backward(loss_t2g)
        nn.utils.clip_grad_norm_(self.t2g_model.parameters(), self.gradient_clip_val)
        t2g_opt.step()

        # --- train g2t one step
        pred, _, kl_div = self.g2t_model(batch_g2t)
        recon_loss = F.nll_loss(
            pred.reshape(-1, pred.shape[-1]),
            batch_g2t["tgt"].reshape(-1),
            ignore_index=0,
        )
        kl_anneal = min(1.0, (self.global_step + 100) / (self.global_step + 10000))
        loss_g2t = recon_loss + kl_anneal * 8.0 / 385 * kl_div
        g2t_opt.zero_grad()
        # todo: combine the 2 loss and avoid manual backward?
        self.manual_backward(loss_g2t)
        # With manual_backward, this needs to be called manually,
        # https://github.com/PyTorchLightning/pytorch-lightning/issues/6328#issuecomment-821655344
        nn.utils.clip_grad_norm_(self.g2t_model.parameters(), self.gradient_clip_val)
        g2t_opt.step()

        # learning rate schedulers
        t2g_sch.step()
        g2t_sch.step()

        self.log_dict(
            {
                "loss_t2g": loss_t2g,
                "loss_g2t": loss_g2t,
                "recon_loss": recon_loss,
                "kl_div": kl_div,
            },
            prog_bar=True,
        )

    def on_validation_epoch_start(self) -> None:
        # should be what we want, compared to on_validation_start
        # https://github.com/PyTorchLightning/pytorch-lightning/issues/2816
        self.wf = open("t2g_out.txt", "w", encoding="utf-8")
        self.g2t_hyp = []
        self.g2t_ref = []
        self.g2t_same = []

    def on_validation_epoch_end(self, outputs) -> None:
        self.wf.close()
        self.logger.experiment.log_artifact(
            "t2g_out.txt", f"t2g_out/{self.current_epoch}"
        )

        # g2t
        self.log_dict(
            {
                "f1_micro": self.f1_micro.compute(),
                "f1_macro": self.f1_macro.compute(),
            }
        )
        # format predictions into text
        unq_hyp = {}
        unq_ref = defaultdict(list)
        self.g2t_hyp = [x[0] for x in self.g2t_hyp]
        self.g2t_ref = [x[0] for x in self.g2t_ref]
        idxs, self.g2t_same = list(
            zip(*sorted(enumerate(self.g2t_same), key=lambda x: x[1]))
        )
        ptr = 0
        for i in range(len(self.g2t_hyp)):
            if i > 0 and self.g2t_same[i] != self.g2t_same[i - 1]:
                ptr += 1
            unq_hyp[ptr] = self.g2t_hyp[idxs[i]]
            unq_ref[ptr].append(self.g2t_ref[idxs[i]])
        max_len = max([len(self.g2t_ref) for self.g2t_ref in unq_ref.values()])
        unq_hyp = sorted(unq_hyp.items(), key=lambda x: x[0])
        unq_ref = sorted(unq_ref.items(), key=lambda x: x[0])
        self.g2t_hyp = [x[1] for x in unq_hyp]
        self.g2t_ref = [[x.lower() for x in y[1]] for y in unq_ref]
        # log predictions
        with open("g2t_out.txt", "w", encoding="utf-8") as wf_h:
            for i, h in enumerate(self.g2t_hyp):
                wf_h.write(str(h) + "\n")
        self.logger.experiment.log_artifact(
            "g2t_out.txt", f"g2t_out/{self.current_epoch}"
        )
        # compute NLG metrics
        g2t_hyp = dict(
            zip(range(len(self.g2t_hyp)), [[x.lower()] for x in self.g2t_hyp])
        )
        g2t_ref = dict(zip(range(len(self.g2t_ref)), self.g2t_ref))
        bleu = Bleu(4)
        meteor = Meteor()
        rouge = Rouge()
        cider = Cider()
        bleu_score = bleu.compute_score(g2t_ref, g2t_hyp)
        self.log_dict(
            {
                "BLEU INP": len(g2t_hyp),
                "BLEU 1": bleu_score[0][0],
                "BLEU 2": bleu_score[0][1],
                "BLEU 3": bleu_score[0][2],
                "BLEU 4": bleu_score[0][3],
                "METEOR": meteor.compute_score(g2t_ref, g2t_hyp)[0],
                "ROUGE_L": (rouge.compute_score(g2t_ref, g2t_hyp)[0]),
                "Cider": cider.compute_score(g2t_ref, g2t_hyp)[0],
            }
        )

    def validation_step(self, batch, batch_idx):
        batch_t2g = batch2tensor_t2g(
            batch, self.device, self.text_vocab, self.ent_vocab, self.tokenizer
        )
        batch_g2t = batch2tensor_g2t(batch, self.device, self.ent_vocab)

        # t2g
        pred = self.t2g_model(batch_t2g)
        _pred = pred.view(-1, pred.shape[-1]).argmax(-1).cpu().long().tolist()
        _gold = batch_t2g["tgt"].view(-1).cpu().long().tolist()
        tpred = pred.argmax(-1).cpu().numpy()
        tgold = batch_t2g["tgt"].cpu().numpy()
        # save predictions as plain text (todo: use custom callback)
        for j in range(len(batch)):
            ents = [
                [y for y in self.ent_vocab(x) if y[0] != "<"]
                for x in batch[j]["ent_text"]
            ]
            self.wf.write("=====================\n")
            self.wf.write("--- Predictions\n")
            self.write_t2g_log(tpred[j], ents)
            self.wf.write("--- Target\n")
            self.write_t2g_log(tgold[j], ents)
        # compute f1 metrics
        pred, gold = [], []
        for j in range(len(_gold)):
            if _gold[j] > 0:  # not the <PAD>
                # todo: simply use ignore_index of F1 metric?
                pred.append(_pred[j])
                gold.append(_gold[j])
        pred = torch.Tensor(pred)
        target = torch.Tensor(gold)
        f1_micro = self.f1_micro(pred, target)
        f1_macro = self.f1_macro(pred, target)

        # g2t
        seq = self.g2t_model(batch_g2t, beam_size=self.beam_size)  # (bs, max_sent_len)
        r = write_txt(batch, batch["tgt"], self.text_vocab)
        h = write_txt(batch, seq, self.text_vocab)
        self.g2t_same.extend(
            [str(x["raw_relation"]) + str(x["ent_text"]) for x in batch]
        )
        # save text predictions for evaluation and logging at the end of val epoch
        self.g2t_hyp.extend(h)
        self.g2t_ref.extend(r)

        # log step-wise metrics
        self.log_dict({"f1_micro_step": f1_micro, "f1_macro_step": f1_macro})

    def test_step(self, *args, **kwargs):
        pass

    def write_t2g_log(self, rel_pred: np.array, ents: List[str]):
        """
        Write a predicted graph as plain text to the logging file

        Args:
            rel_pred: numpy array of shape (max_num_ent, max_num_ent).
                Contains the indices of predicted relations between entities.
            ents: list of entities as plain text, for this example

        """
        rels = []
        for e1 in range(len(ents)):
            for e2 in range(len(ents)):
                # 0: padding, 3: unknown
                if rel_pred[e1, e2] != 3 and rel_pred[e1, e2] != 0:
                    rels.append((e1, int(rel_pred[e1, e2]), e2))
        self.wf.write(
            str([(ents[e1], self.rel_vocab(r), ents[e2]) for e1, r, e2 in rels]) + "\n"
        )
