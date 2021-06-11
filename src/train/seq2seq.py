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
from src.utils import (
    seed_everything,
    mlflow_log_src_and_config,
    ModelSummary,
)


class EvalCallback(TrainerCallback):
    """
    Run evaluation on the specified dataset on epoch end.
    Also log predictions as txt files in mlflow
    """

    def __init__(
        self,
        log_path: Path,
        tb_writer: SummaryWriter,
        model,
        dataset: WebNLG,
        batch_size,
        num_beams,
    ):
        self.log_path = log_path
        self.tb_writer = tb_writer
        self.model = model
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_beams = num_beams

    def evaluate(self, epoch: int):
        epoch = int(epoch)
        self.model.eval()  # .no_grad() already called by model.generate(), but not .eval()
        device = torch.device(0 if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        res, logs = self.dataset.evaluate_dataset(
            self.model,
            device=device,
            batch_size=self.batch_size,
            num_beams=self.num_beams,
        )
        # print and save eval metrics
        logging.info(f"[ep{epoch}] eval results: {res}")
        mlflow.log_metrics(res, step=epoch)
        for k, v in res.items():
            self.tb_writer.add_scalar(k, v, global_step=epoch)

        # save predictions logs to mlflow
        with open(self.log_path / f"t2g_{epoch}.txt", "w", encoding="utf-8") as f:
            f.write(logs)
        mlflow.log_artifact(str(self.log_path / f"t2g_{epoch}.txt"), f"t2g_out/{epoch}")

    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        self.evaluate(state.epoch)


def main(timestamp: str):
    # Load config
    project_dir = Path(__file__).resolve().parents[2]
    conf = OmegaConf.load(project_dir / "conf/conf_seq_to_seq.yaml")
    logging.info(OmegaConf.to_yaml(conf))

    # seed everything
    seed_everything(conf.seed)

    mlflow.set_tracking_uri("https://mlflow.par.prod.crto.in/")
    mlflow.set_experiment("al.thomas_data_2_text")
    run_name = f"{timestamp}-{'sup' if conf.supervised else 'unsup'}-{conf.model}"
    tb_writer = SummaryWriter(log_dir=str(project_dir / f"models/{run_name}"))
    logging.info(f"run_name: {run_name}\n")

    with mlflow.start_run(run_name=run_name):
        mlflow_log_src_and_config(conf, project_dir)

        # load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(conf.model)
        # add as regular tokens (not as special tokens like <pad> since we need to
        tokenizer.add_tokens(
            [
                OutputFormat.HEAD_TOKEN,
                OutputFormat.TYPE_TOKEN,
                OutputFormat.TAIL_TOKEN,
            ]
        )

        # load data
        dataset_train = WebNLG(
            data_dir=project_dir / "data", split="train", tokenizer=tokenizer
        )
        dataset_val = WebNLG(
            data_dir=project_dir / "data",
            split="val",
            tokenizer=tokenizer,
            limit_samples=100,
        )
        dataset_test = WebNLG(
            data_dir=project_dir / "data", split="test", tokenizer=tokenizer
        )

        # prepare model
        model = T5ForConditionalGeneration.from_pretrained(conf.model)
        # extend embedding matrices to include our separator tokens
        model.resize_token_embeddings(len(tokenizer))
        summary = ModelSummary(model, mode="top")
        logging.info(f"\n{summary}")

        val_callback = EvalCallback(
            project_dir / f"models/{run_name}",
            tb_writer,
            model,
            dataset_val,
            conf.batch_size_val,
            conf.num_beams,
        )
        test_callback = EvalCallback(
            project_dir / f"models/{run_name}",
            tb_writer,
            model,
            dataset_test,
            conf.batch_size_val,
            conf.num_beams,
        )
        val_callback.evaluate(-1)

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
            train_dataset=dataset_train,
            eval_dataset=dataset_val,
            callbacks=[val_callback],
        )
        trainer.remove_callback(MLflowCallback)
        trainer.train()

        # test
        test_callback.evaluate(0)
