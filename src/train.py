import logging
from pathlib import Path

import mlflow
import torch
import torch.optim
import torch.optim
from omegaconf import OmegaConf
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from transformers import (
    T5ForConditionalGeneration,
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    TrainerCallback,
    TrainerState,
    TrainerControl,
)
from transformers.integrations import MLflowCallback

from src.data.datasets import WebNLG
from src.data.formatting import OutputFormat
from src.eval import EvalCallback, Evaluator
from src.utils import (
    seed_everything,
    mlflow_log_src_and_config,
    ModelSummary,
)


def main(timestamp: str):
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
            # limit_samples=10,
        )
        eval_callback = EvalCallback(evaluator)

        # train model
        training_args = TrainingArguments(
            output_dir=tb_writer.log_dir,
            learning_rate=conf.lr,
            per_device_train_batch_size=conf.batch_size_train,
            logging_dir=tb_writer.log_dir,
            save_strategy="no",
            #  To ensure reproducibility across runs, use the model_init() function
            #  to instantiate the model if it has some randomly initialized parameters?
            seed=conf.seed,
            num_train_epochs=conf.epochs,
            # max_steps=5,
        )
        logging.info("training...")
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            callbacks=[eval_callback],
        )
        trainer.remove_callback(MLflowCallback)
        trainer.train()

        # save model checkpoint
        if conf.checkpoint:
            torch.save(model.state_dict(), f"/tmp/{run_name}_t2g_{conf.model}.pt")
            mlflow.log_artifact(
                f"/tmp/{run_name}_t2g_{conf.model}.pt", f"t2g_{conf.model}.pt"
            )
