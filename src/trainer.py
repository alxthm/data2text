import logging

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import (
    AdamW,
    get_scheduler,
    default_data_collator,
)

from src.data.datasets import Seq2seqDataset
from src.eval.evaluator import EvaluatorWebNLG
from src.utils import MyLogger, Mode


class Seq2seqTrainer:
    def __init__(
        self,
        model,
        mode: Mode,
        train_dataset: Seq2seqDataset,
        evaluator: EvaluatorWebNLG,
        learning_rate: float,
        batch_size: int,
        num_epochs: int,
        tensorboard_writer: SummaryWriter,
        log_every_n_steps: int,
        max_training_steps: int = -1,
    ):
        self.model = model
        self.mode = mode
        self.device = model.device

        # training
        self.optimizer = AdamW(model.parameters(), lr=learning_rate)
        self.train_dataloader = DataLoader(
            train_dataset,
            shuffle=True,
            batch_size=batch_size,
            collate_fn=default_data_collator,
        )
        if max_training_steps > 0:
            self.num_training_steps = max_training_steps
            self.num_epochs = 1
        else:
            self.num_training_steps = num_epochs * len(self.train_dataloader)
            self.num_epochs = num_epochs
        self.lr_scheduler = get_scheduler(
            "linear",
            optimizer=self.optimizer,
            num_warmup_steps=0,
            num_training_steps=self.num_training_steps,
        )
        self.batch_size = batch_size
        self.max_output_length = train_dataset.max_seq_length

        # eval
        self.evaluator = evaluator

        # logging
        self.logger = MyLogger(
            tensorboard_writer=tensorboard_writer, log_every_n_steps=log_every_n_steps
        )

    def teach_t2g_one_step(self, text_ids, att_mask_text, graph_ids):
        """

        Args:
            text_ids: input text (tokenized batch, already on device)
            att_mask_text: input attention mask (for padding)
            graph_ids: label (ground truth graph as a tokenized sequence)

        Returns:
            loss_t2g

        """
        self.model.train()
        t2g_outputs = self.model(
            input_ids=text_ids,
            attention_mask=att_mask_text,
            labels=graph_ids,
        )
        loss_t2g = t2g_outputs.loss
        loss_t2g.backward()
        return loss_t2g

    def teach_g2t_one_step(self, graph_ids, att_mask_graph, text_ids):
        """

        Args:
            graph_ids: input graph (tokenized batch, already on device)
            att_mask_graph: input attention mask (for padding)
            text_ids: label text (tokenized as well)

        Returns:
            loss_g2t

        """
        self.model.train()
        g2t_outputs = self.model(
            input_ids=graph_ids,
            attention_mask=att_mask_graph,
            labels=text_ids,
        )
        loss_g2t = g2t_outputs.loss
        loss_g2t.backward()
        return loss_g2t

    def predict_graph(self, text_ids):
        self.model.eval()
        with torch.no_grad():
            graph_pred_ids = self.model.generate(
                text_ids,
                max_length=self.max_output_length,
                num_beams=1,
            )
            # attention mask: 0 if it's a padding token, 1 otherwise
            graph_pred_att_mask = (graph_pred_ids != 0).type_as(graph_pred_ids)
        return graph_pred_ids, graph_pred_att_mask

    def predict_text(self, graph_ids):
        self.model.eval()
        with torch.no_grad():
            text_pred_ids = self.model.generate(
                graph_ids,
                max_length=self.max_output_length,
                num_beams=1,
            )
            # attention mask: 0 if it's a padding token, 1 otherwise
            text_pred_att_mask = (text_pred_ids != 0).type_as(text_pred_ids)
        return text_pred_ids, text_pred_att_mask

    def train(self):
        global_step = 0
        logging.info("Training...")
        logging.info(f"     num_epochs: {self.num_epochs}")

        for epoch in range(self.num_epochs):
            for batch in tqdm(self.train_dataloader, desc=f"[ep{epoch}]"):
                # stop training if a max number of steps was specified
                if global_step > self.num_training_steps:
                    break

                # move data to device
                text_ids = batch["text_ids"].to(self.device)
                att_mask_text = batch["att_mask_text"].to(self.device)
                graph_ids = batch["graph_ids"].to(self.device)
                att_mask_graph = batch["att_mask_graph"].to(self.device)

                # training step
                if self.mode == Mode.t2g:
                    loss_g2t = torch.tensor(0)
                    loss_t2g = self.teach_t2g_one_step(
                        text_ids=text_ids,
                        att_mask_text=att_mask_text,
                        graph_ids=graph_ids,
                    )
                elif self.mode == Mode.g2t:
                    loss_g2t = self.teach_g2t_one_step(
                        graph_ids=graph_ids,
                        att_mask_graph=att_mask_graph,
                        text_ids=text_ids,
                    )
                    loss_t2g = torch.tensor(0)
                elif self.mode == Mode.both_sup:
                    loss_g2t = self.teach_g2t_one_step(
                        graph_ids=graph_ids,
                        att_mask_graph=att_mask_graph,
                        text_ids=text_ids,
                    )
                    loss_t2g = self.teach_t2g_one_step(
                        text_ids=text_ids,
                        att_mask_text=att_mask_text,
                        graph_ids=graph_ids,
                    )
                elif self.mode == Mode.both_unsup:
                    # todo: shuffle the training dataset
                    # todo: make sure the predictions are correctly formatted
                    #   (e.g. is the attention mask ok? does it start with a padding token and why?)
                    # g2t unsupervised step
                    graph_pred_ids, att_mask_pred_graph = self.predict_graph(
                        text_ids=text_ids
                    )
                    loss_g2t = self.teach_g2t_one_step(
                        graph_ids=graph_pred_ids,
                        att_mask_graph=att_mask_pred_graph,
                        text_ids=text_ids,
                    )
                    # t2g unsupervised step
                    text_pred_ids, att_mask_pred_text = self.predict_text(
                        graph_ids=graph_ids
                    )
                    loss_t2g = self.teach_t2g_one_step(
                        text_ids=text_pred_ids,
                        att_mask_text=att_mask_pred_text,
                        graph_ids=graph_ids,
                    )
                else:
                    raise ValueError

                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
                global_step += 1

                # log training info
                self.logger.log_metrics(
                    {
                        "train/loss_t2g": loss_t2g.item(),
                        "train/loss_g2t": loss_g2t.item(),
                        "train/learning_rate": self.lr_scheduler.get_last_lr()[0],
                        "train/epoch": epoch,
                    },
                    step=global_step,
                )

            # evaluate after each epoch
            self.evaluator.evaluate_dev(epoch)
