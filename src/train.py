import logging
from pathlib import Path

import mlflow
import torch
import torch.optim
import torch.optim
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import (
    T5ForConditionalGeneration,
    AutoTokenizer,
    AdamW,
    get_scheduler,
)

from src.data.datasets import WebNLG
from src.data.formatting import OutputFormat
from src.eval import EvalCallback, Evaluator
from src.utils import (
    seed_everything,
    mlflow_log_src_and_config,
    ModelSummary,
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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # training
        self.optimizer = AdamW(model.parameters(), lr=learning_rate)
        self.train_dataloader = DataLoader(
            train_dataset, shuffle=True, batch_size=self.batch_size
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
        self.logger = MyLogger(tensorboard_writer)

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
                        "learning_rate": self.lr_scheduler.get_last_lr()[0],
                        "epoch": epoch,
                    },
                    step=global_step,
                )

            # evaluate after each epoch
            self.evaluator.evaluate_and_log(epoch, "val")

        # final evaluation, on test set
        self.evaluator.evaluate_and_log(0, "test")


def train(timestamp: str):
    # Load config
    project_dir = Path(__file__).resolve().parents[1]
    conf = OmegaConf.load(project_dir / "conf/conf_seq_to_seq.yaml")
    logging.info(OmegaConf.to_yaml(conf))

    # seed everything
    seed_everything(conf.seed)

    mlflow.set_tracking_uri("https://mlflow.par.prod.crto.in/")
    mlflow.set_experiment("al.thomas_d2t_3")
    run_name = f"{timestamp}-{'sup' if conf.supervised else 'unsup'}-{conf.model}"
    tb_writer = SummaryWriter(log_dir=str(project_dir / f"models/{run_name}"))
    logging.info(f"run_name: {run_name}\n")

    with mlflow.start_run(run_name=run_name):
        mlflow_log_src_and_config(conf, project_dir)

        # load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(conf.model)
        # add our separators as new tokens
        # (regular, not special tokens like <pad> since we need them after cleaning output sentence)
        tokenizer.add_tokens(
            [
                OutputFormat.HEAD_TOKEN,
                OutputFormat.TYPE_TOKEN,
                OutputFormat.TAIL_TOKEN,
            ]
        )

        # load data
        data_dir = project_dir / "data"
        train_dataset = WebNLG(data_dir=data_dir, split="train", tokenizer=tokenizer)
        val_dataset = WebNLG(data_dir=data_dir, split="val", tokenizer=tokenizer)
        test_dataset = WebNLG(data_dir=data_dir, split="test", tokenizer=tokenizer)

        # prepare model
        model = T5ForConditionalGeneration.from_pretrained(conf.model)
        # extend embedding matrices to include our separator tokens
        model.resize_token_embeddings(len(tokenizer))
        summary = ModelSummary(model, mode="top")
        logging.info(f"\n{summary}")

        evaluator = Evaluator(
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            tokenizer=tokenizer,
            model=model,
            batch_size=conf.batch_size_val,
            num_beams=conf.num_beams,
            log_path=project_dir / f"models/{run_name}",
            tensorboard_writer=tb_writer,
            limit_samples=10,
        )

        # train model
        trainer = Seq2seqTrainer(
            model=model,
            train_dataset=train_dataset,
            evaluator=evaluator,
            learning_rate=conf.lr,
            batch_size=conf.batch_size_train,
            num_epochs=conf.epochs,
            tensorboard_writer=tb_writer,
            max_training_steps=10,
        )
        trainer.train()

        # save model checkpoint
        if conf.checkpoint:
            torch.save(model.state_dict(), f"/tmp/{run_name}_t2g_{conf.model}.pt")
            mlflow.log_artifact(
                f"/tmp/{run_name}_t2g_{conf.model}.pt", f"t2g_{conf.model}.pt"
            )
