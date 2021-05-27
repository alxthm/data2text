from collections import defaultdict
from typing import List

import mlflow
import numpy as np
import torch
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from transformers import get_cosine_schedule_with_warmup
import torch.nn.functional as F

from src.data.webnlg import Vocab, write_txt
from src.model.g2t import G2T
from src.model.t2g import T2G


class CycleCVAE(nn.Module):
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
        beam_max_len: int,
        length_penalty: float,
        t2g_drop: float,
        t2g_lr: float,
        t2g_weight_decay: float,
        gradient_clip_val: float,
        run_name: str,
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
        self.run_name = run_name

        # category weights: give less importance to more frequent relations
        t2g_weight = [rel_vocab.wf.get(x, 0) for x in rel_vocab.i2s]
        t2g_weight[0] = 0
        max_w = max(t2g_weight)
        t2g_weight = np.array(t2g_weight).astype("float32")
        t2g_weight = (max_w + 1000) / (t2g_weight + 1000)
        # make it part of state_dict, and move to correct device with parameters
        self.register_buffer("t2g_weight", torch.from_numpy(t2g_weight).float())

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
            beam_max_len=beam_max_len,
            length_penalty=length_penalty,
        )
        self.t2g_model = T2G(
            relation_types=len(rel_vocab), d_model=dim_h, dropout=t2g_drop
        )

        # optimizers
        self.g2t_optimizer = torch.optim.Adam(
            self.g2t_model.parameters(),
            lr=self.g2t_lr,
            weight_decay=self.g2t_weight_decay,
        )
        self.t2g_optimizer = torch.optim.Adam(
            self.t2g_model.parameters(),
            lr=self.t2g_lr,
            weight_decay=self.t2g_weight_decay,
        )

        # learning rate schedulers
        self.g2t_scheduler = get_cosine_schedule_with_warmup(
            optimizer=self.g2t_optimizer,
            num_warmup_steps=400,
            num_training_steps=800 * self.tot_epoch,
        )
        self.t2g_scheduler = get_cosine_schedule_with_warmup(
            optimizer=self.t2g_optimizer,
            num_warmup_steps=400,
            num_training_steps=800 * self.tot_epoch,
        )

        # metrics
        self.bleu = Bleu(4)
        self.meteor = Meteor()
        self.rouge = Rouge()
        self.cider = Cider()

    def training_step(self, batch, global_step: int):
        # --- SUPERVISED
        self.train()

        # --- train t2g one step
        pred = self.t2g_model(batch)  # pred (bs, ne, ne, num_relations)
        loss_t2g = F.nll_loss(
            pred.contiguous().view(-1, pred.shape[-1]),
            batch["t2g_tgt"].contiguous().view(-1),  # initially shape (bs, ne, ne)
            ignore_index=0,
            weight=self.t2g_weight,
        )
        self.t2g_optimizer.zero_grad()
        loss_t2g.backward()
        nn.utils.clip_grad_norm_(self.t2g_model.parameters(), self.gradient_clip_val)
        self.t2g_optimizer.step()

        # --- train g2t one step
        pred, _, kl_div = self.g2t_model(batch)
        recon_loss = F.nll_loss(
            pred.reshape(-1, pred.shape[-1]),
            batch["g2t_tgt"].reshape(-1),
            ignore_index=0,
        )
        kl_anneal = min(1.0, (global_step + 100) / (global_step + 10000))
        loss_g2t = recon_loss + kl_anneal * 8.0 / 385 * kl_div
        self.g2t_optimizer.zero_grad()
        loss_g2t.backward()
        nn.utils.clip_grad_norm_(self.g2t_model.parameters(), self.gradient_clip_val)
        self.g2t_optimizer.step()

        # learning rate schedulers
        self.t2g_scheduler.step()
        self.g2t_scheduler.step()

        return {
            "train_loss_t2g": loss_t2g.item(),
            "train_loss_g2t": loss_g2t.item(),
            "train_recon_loss": recon_loss.item(),
            "train_kl_div": kl_div.item(),
        }

    def on_eval_epoch_start(self):
        self.wf_t2g = open(f"/tmp/{self.run_name}_t2g_out.txt", "w", encoding="utf-8")
        self.t2g_hyp = []
        self.t2g_ref = []
        self.g2t_hyp = []
        self.g2t_ref = []
        self.g2t_same = []

    def eval_step(self, batch):
        batch_size = len(batch["original_ent_text"])

        # t2g
        t2g_logits = self.t2g_model(batch)  # (bs, ne, ne, num_relations)
        num_relations = t2g_logits.shape[-1]
        self.wf_t2g.write(
            self.graph_preds_as_str(
                t2g_logits, batch["t2g_tgt"], batch["original_ent_text"]
            )
        )
        # save predictions to later compute f1 metrics
        self.t2g_hyp.append(t2g_logits.view(-1, num_relations).argmax(-1).cpu())
        self.t2g_ref.append(batch["t2g_tgt"].view(-1).cpu())

        # g2t
        seq = self.g2t_model(batch, beam_size=self.beam_size)  # (bs, max_sent_len)
        r = write_txt(batch, batch["g2t_tgt"], self.text_vocab)
        h = write_txt(batch, seq, self.text_vocab)
        self.g2t_same += [
            str(batch["original_raw_relation"][i]) + str(batch["original_ent_text"][i])
            for i in range(batch_size)
        ]

        # save text predictions for evaluation and logging at the end of val epoch
        self.g2t_hyp += h
        self.g2t_ref += r

    def on_eval_epoch_end(self, tag: str, tb_writer: SummaryWriter, global_step: int):
        # compute metrics over all validation batch and log results
        self.wf_t2g.close()
        preds = torch.cat(self.t2g_hyp, dim=0).numpy()
        target = torch.cat(self.t2g_ref, dim=0).numpy()
        # labels that we want to consider
        # (ignore targets that correspond to padding (PAD_ID=0) or no relation (UNK_ID=3))
        labels = list(set(target.tolist()))
        labels = [x for x in labels if x not in [0, 3]]
        # micro: f1 score of all predictions, no matter the label
        f1_micro = f1_score(
            preds,
            target,
            average="micro",
            labels=labels,
            zero_division=0,
        )
        # macro: average of f1 scores for all labels
        f1_macro = f1_score(
            preds,
            target,
            average="macro",
            labels=labels,
            zero_division=0,
        )
        mlflow.log_artifact(
            f"/tmp/{self.run_name}_t2g_out.txt", f"t2g_out/{global_step}"
        )
        mlflow.log_metrics(
            {f"{tag}_f1_micro": f1_micro, f"{tag}_f1_macro": f1_macro}, step=global_step
        )
        tb_writer.add_scalar(f"{tag}_f1_micro", f1_micro, global_step=global_step)
        tb_writer.add_scalar(f"{tag}_f1_macro", f1_macro, global_step=global_step)

        # g2t - format predictions into text
        unq_hyp = {}
        unq_ref = defaultdict(list)
        g2t_hyp = [x[0] for x in self.g2t_hyp]
        g2t_ref = [x[0] for x in self.g2t_ref]
        idxs, g2t_same = list(
            zip(*sorted(enumerate(self.g2t_same), key=lambda x: x[1]))
        )
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
        with open(f"/tmp/{self.run_name}_g2t_out.txt", "w", encoding="utf-8") as wf_g2t:
            for i in range(len(g2t_hyp)):
                wf_g2t.write(
                    f"{i}\n" f"pred: {str(g2t_hyp[i])}\n" f"tgt: {str(g2t_ref[i])}\n"
                )
        mlflow.log_artifact(
            f"/tmp/{self.run_name}_g2t_out.txt", f"g2t_out/{global_step}"
        )
        # compute NLG metrics
        g2t_hyp = dict(zip(range(len(g2t_hyp)), [[x.lower()] for x in g2t_hyp]))
        g2t_ref = dict(zip(range(len(g2t_ref)), g2t_ref))
        bleu_score = self.bleu.compute_score(g2t_ref, g2t_hyp)
        metrics = {
            f"{tag}_bleu_1": bleu_score[0][0],
            f"{tag}_bleu_2": bleu_score[0][1],
            f"{tag}_bleu_3": bleu_score[0][2],
            f"{tag}_bleu_4": bleu_score[0][3],
            f"{tag}_meteor": self.meteor.compute_score(g2t_ref, g2t_hyp)[0],
            f"{tag}_rouge_l": (self.rouge.compute_score(g2t_ref, g2t_hyp)[0]),
            f"{tag}_cider": self.cider.compute_score(g2t_ref, g2t_hyp)[0],
        }
        mlflow.log_metrics(metrics, step=global_step)
        for k, v in metrics.items():
            tb_writer.add_scalar(k, v, global_step=global_step)

    def log_gradients(
        self, tb_writer: SummaryWriter, global_step: int, print_warning: bool
    ):
        # log histogram for weights and gradients
        for name, param in self.named_parameters():
            tb_writer.add_histogram(name, param, global_step)
            if param.requires_grad:
                try:
                    tb_writer.add_histogram(f"{name}_grad", param.grad, global_step)
                except NotImplementedError:
                    if print_warning:
                        print(f"[{global_step}] No gradient for param {name}")

    def graph_preds_as_str(
        self, pred: torch.FloatTensor, target: torch.LongTensor, original_ent_text
    ):
        """

        Args:
            pred: logits, shape (bs, ne, ne, num_rel) where ne is the max number
                of entities in a batch sentence
            target: shape (bs, ne, ne)

        Returns:

        """
        batch_size, _, _, _ = pred.shape
        tpred = pred.argmax(-1).cpu().numpy()
        tgold = target.cpu().numpy()
        s = ""

        def graph_as_str(relations: np.array, ents: List[List]):
            """
            Write a predicted graph as plain text to the logging file

            Args:
                relations: numpy array of shape (max_num_ent, max_num_ent).
                    Contains the indices of relations between entities.
                ents: list of entities for this example, each entity being a list of tokens

            """
            rels = []
            for e1 in range(len(ents)):
                for e2 in range(len(ents)):
                    # 0: padding, 3: unknown
                    if relations[e1, e2] != 3 and relations[e1, e2] != 0:
                        rels.append((e1, int(relations[e1, e2]), e2))
            linearized_graph = [
                (ents[e1], self.rel_vocab(r), ents[e2]) for e1, r, e2 in rels
            ]
            return f"{str(linearized_graph)}\n"

        for j in range(batch_size):
            ents = [
                [y for y in self.ent_vocab(x) if y[0] != "<"]
                for x in original_ent_text[j]
            ]
            s += "=====================\n"
            s += "--- Predictions\n"
            s += graph_as_str(relations=tpred[j], ents=ents)
            s += "--- Target\n"
            s += graph_as_str(relations=tgold[j], ents=ents)
        return s
