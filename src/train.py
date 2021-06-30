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
from src.data.formatting import Mode
from src.eval.evaluator import EvaluatorWebNLG
from src.utils import (
    MyLogger,
)


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

        # eval
        self.evaluator = evaluator

        # logging
        self.logger = MyLogger(tensorboard_writer=tensorboard_writer)

    def train(self):
        global_step = 0
        logging.info("Training...")
        logging.info(f"     num_epochs: {self.num_epochs}")

        self.model.train()
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
                    t2g_outputs = self.model(
                        input_ids=text_ids,
                        attention_mask=att_mask_text,
                        labels=graph_ids,
                    )
                    loss_t2g = t2g_outputs.loss
                    loss_t2g.backward()
                elif self.mode == Mode.g2t:
                    loss_t2g = torch.tensor(0)
                    g2t_outputs = self.model(
                        input_ids=graph_ids,
                        attention_mask=att_mask_graph,
                        labels=text_ids,
                    )
                    loss_g2t = g2t_outputs.loss
                    loss_g2t.backward()
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

