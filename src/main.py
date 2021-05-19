import glob
import os
from datetime import datetime
from pathlib import Path

import mlflow
import numpy as np
import pytorch_lightning as pl
import shutil

import random

import torch
from omegaconf import OmegaConf
from pytorch_lightning.loggers import MLFlowLogger

from src.data.webnlg import WebNLGDataModule
from src.model.cvae import CycleCVAE


def save_src(log_dir: str, conf):
    os.makedirs(log_dir, exist_ok=True)
    # save config and source files as text files
    with open(f"{log_dir}/conf.yaml", "w") as f:
        OmegaConf.save(conf, f)
    for f in glob.iglob("*.py"):
        shutil.copy2(f, log_dir)


def main():
    # Load config
    project_dir = Path(__file__).resolve().parents[1]
    conf = OmegaConf.load(project_dir / "conf/conf.yaml")
    print(OmegaConf.to_yaml(conf))

    # seed everything # todo: use pl?
    random.seed(conf.seed)
    torch.manual_seed(conf.seed)
    np.random.seed(conf.seed)
    torch.cuda.manual_seed_all(conf.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Save and log config/code
    mlf_logger = MLFlowLogger(
        experiment_name="al.thomas_data_2_text",
        tracking_uri="https://mlflow.par.prod.crto.in/",
        tags={"mode": conf.mode},
    )
    mlf_logger.log_hyperparams(OmegaConf.to_container(conf))
    for f in (project_dir / "src").rglob("*.py"):
        mlf_logger.experiment.log_artifact(mlf_logger.run_id, f)

    # create/load data and vocab
    webnlg = WebNLGDataModule(
        data_dir=project_dir / "data/", batch_size=conf.batch_size
    )
    # https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html#using-a-datamodule
    webnlg.prepare_data()
    webnlg.setup()
    # save everything to mlflow
    for path in [
        webnlg.dataset_train_path,
        webnlg.dataset_val_path,
        webnlg.dataset_test_path,
        webnlg.vocab_path,
    ]:
        mlf_logger.experiment.log_artifact(mlf_logger.run_id, path)

    # train
    # todo: log checkpoints to mlflow as well
    model = CycleCVAE(
        text_vocab=webnlg.text_vocab,
        ent_vocab=webnlg.ent_vocab,
        rel_vocab=webnlg.rel_vocab,
        tokenizer=webnlg.tokenizer,
        tot_epoch=conf.tot_epoch,
        dim_h=conf.dim_h,
        dim_z=conf.dim_z,
        enc_lstm_layers=conf.g2t.enc_lstm_layers,
        n_head=conf.g2t.n_head,
        head_dim=conf.g2t.head_dim,
        emb_dropout=conf.g2t.emb_dropout,
        attn_drop=conf.g2t.attn_drop,
        drop=conf.g2t.drop,
        n_layers_gat=conf.g2t.n_layers_gat,
        g2t_lr=conf.g2t.lr,
        g2t_weight_decay=conf.g2t.weight_decay,
        beam_size=conf.g2t.beam_size,
        beam_max_len=conf.g2t.beam_max_len,
        length_penalty=conf.g2t.length_penalty,
        t2g_drop=conf.t2g.drop,
        t2g_lr=conf.t2g.lr,
        t2g_weight_decay=conf.t2g.weight_decay,
        gradient_clip_val=conf.grad_clip,
    )
    # todo: test if lightning gives an error for manual optim + clip grad val
    #   then comment on https://github.com/PyTorchLightning/pytorch-lightning/issues/6328
    #   to request improving documentation since this PR removed the hint
    #   (https://github.com/PyTorchLightning/pytorch-lightning/pull/6907)
    trainer = pl.Trainer(logger=mlf_logger, max_epochs=conf.tot_epoch, gpus=1)
    trainer.fit(model, datamodule=webnlg)

    trainer.test(datamodule=webnlg)


if __name__ == "__main__":
    main()
