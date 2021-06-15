import logging
from collections import Counter

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import (
    AdamW,
    get_scheduler,
    default_data_collator,
)

from src.eval import Evaluator
from src.utils import (
    MyLogger,
)


class Seq2seqTrainer:
    def __init__(
        self,
        model,
        train_dataset,
        evaluator: Evaluator,
        learning_rate: float,
        batch_size: int,
        num_epochs: int,
        tensorboard_writer: SummaryWriter,
        max_training_steps: int = -1,
    ):
        self.model = model
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
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                # training step
                outputs = self.model(
                    input_ids, attention_mask=attention_mask, labels=labels
                )
                loss = outputs.loss
                loss.backward()

                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
                global_step += 1

                # log training info
                self.logger.log_metrics(
                    {
                        "train/loss": loss.item(),
                        "train/learning_rate": self.lr_scheduler.get_last_lr()[0],
                        "train/epoch": epoch,
                    },
                    step=global_step,
                )

            # evaluate after each epoch
            self.evaluator.evaluate_and_log(epoch, "val")

        # final evaluation, on test set
        self.evaluator.evaluate_and_log(0, "test")
