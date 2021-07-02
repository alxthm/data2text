import datetime
import logging
import sys
from pathlib import Path

import mlflow
import torch
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, T5ForConditionalGeneration

from src.data.datasets import WebNLG2020
from src.data.formatting import GraphFormat
from src.eval.evaluator import EvaluatorWebNLG
from src.trainer import Seq2seqTrainer
from src.utils import (
    WarningsFilter,
    seed_everything,
    mlflow_log_src_and_config,
    ModelSummary,
    Mode,
)

logging.basicConfig(
    format="%(asctime)s %(message)s", datefmt="[%H:%M:%S]", level=logging.INFO
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
    run_name = f"{timestamp}-{conf.mode}-{conf.model}"
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
                GraphFormat.HEAD_TOKEN,
                GraphFormat.TYPE_TOKEN,
                GraphFormat.TAIL_TOKEN,
            ]
        )

        # load data
        data_dir = project_dir / "data"
        datasets = {
            split: WebNLG2020(data_dir=data_dir, split=split, tokenizer=tokenizer)
            for split in WebNLG2020.splits
        }
        train_dataset = datasets["train"]

        # prepare model
        model = T5ForConditionalGeneration.from_pretrained(conf.model)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        # extend embedding matrices to include our separator tokens
        model.resize_token_embeddings(len(tokenizer))
        summary = ModelSummary(model, mode="top")
        logging.info(f"\n{summary}")

        evaluator = EvaluatorWebNLG(
            run_name=run_name,
            mode=Mode(conf.mode),
            datasets=datasets,
            tokenizer=tokenizer,
            model=model,
            batch_size=conf.batch_size_val,
            num_beams_t2g=conf.num_beams_t2g,
            num_beams_g2t=conf.num_beams_g2t,
            log_path=project_dir / f"models/{run_name}",
            checkpoints=conf.checkpoints,
            tensorboard_writer=tb_writer,
            limit_samples=10 if conf.fast_dev_run else False,
        )

        # train model
        trainer = Seq2seqTrainer(
            model=model,
            mode=Mode(conf.mode),
            train_dataset=train_dataset,
            evaluator=evaluator,
            learning_rate=conf.lr,
            batch_size=conf.batch_size_train,
            num_epochs=conf.epochs,
            tensorboard_writer=tb_writer,
            log_every_n_steps=conf.log_every_n_steps,
            max_training_steps=10 if conf.fast_dev_run else -1,
        )
        trainer.train()


if __name__ == "__main__":
    # filter out some hdfs warnings (from 3rd party python libraries)
    sys.stdout = WarningsFilter(sys.stdout)
    sys.stderr = WarningsFilter(sys.stderr)
    main(timestamp=datetime.datetime.today().strftime("%m%d%H%M%S"))
