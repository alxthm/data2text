import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from transformers import get_cosine_schedule_with_warmup
import torch.nn.functional as F

from src.data.webnlg import Vocab
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


    def training_step(self, batch, global_step: int):
        # --- SUPERVISED
        self.g2t_model.blind, self.t2g_model.blind = False, False  # not necessary
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

    def validation_step(self, batch):
        pass

    def on_train_batch_end(self, tb_writer: SummaryWriter, global_step: int):
        # log histogram for weights and gradients
        if global_step % 50 == 0:
            for name, param in self.named_parameters():
                tb_writer.add_histogram(name, param, global_step)
                if param.requires_grad:
                    try:
                        tb_writer.add_histogram(
                            f"{name}_grad", param.grad, global_step
                        )
                    except NotImplementedError:
                        print(f"[{global_step}] No gradient for param {name}")
