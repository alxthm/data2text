import datetime
import datetime
import random
import sys
from pathlib import Path

import mlflow
import numpy as np
import torch
import torch.optim
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.data.webnlg import prepare_data, tokenizer
from src.model.cycle_cvae import CycleCVAE
from src.utils import WarningsFilter

tb_writer: SummaryWriter = None
run_name = ""


def train(conf, device, dataloader_train, dataloader_val, dataloader_test, vocab):
    print("Preparing model...")
    model = CycleCVAE(
        text_vocab=vocab["text"],
        ent_vocab=vocab["entity"],
        rel_vocab=vocab["relation"],
        tokenizer=tokenizer,  # todo: obtain correctly instead of importing
        tot_epoch=conf.main.epoch,
        dim_h=conf.main.dim_h,
        dim_z=conf.g2t.vae_dim,
        enc_lstm_layers=conf.g2t.enc_lstm_layers,
        n_head=conf.g2t.nhead,
        head_dim=conf.g2t.head_dim,
        emb_dropout=conf.g2t.emb_drop,
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
        gradient_clip_val=conf.main.grad_clip,
        run_name=run_name,
    )
    model.to(device)

    global_step = 0
    for ep in range(0, conf.main.epoch):
        # training epoch
        for batch in tqdm(dataloader_train, desc=f"[train][ep{ep}]"):
            metrics = model.training_step(batch, global_step=global_step)
            model.on_train_step_end(tb_writer, global_step)
            mlflow.log_metrics(metrics, step=global_step)
            for k, v in metrics.items():
                tb_writer.add_scalar(k, v, global_step=global_step)
            global_step += 1

        # validation epoch
        model.eval()
        with torch.no_grad():
            model.on_eval_epoch_start()
            for batch in tqdm(dataloader_val, desc=f"[val][ep{ep}]"):
                model.eval_step(batch)
            model.on_eval_epoch_end("val", tb_writer, global_step)

            if ep % 15 == 0 and conf.main.save_checkpoints:
                # save model checkpoint
                torch.save(model.state_dict(), f"/tmp/{run_name}_model_ep{ep}.pt")
                mlflow.log_artifact(
                    f"/tmp/{run_name}_model_ep{ep}.pt", f"model_ep{ep}.pt"
                )

    # test
    torch.save(model.state_dict(), f"/tmp/{run_name}_model_final.pt")
    mlflow.log_artifact(f"/tmp/{run_name}_model_final.pt", "model_final.pt")
    model.eval()
    with torch.no_grad():
        model.on_eval_epoch_start()
        for batch in tqdm(dataloader_test, desc="[test]"):
            model.eval_step(batch)
        model.on_eval_epoch_end("test", tb_writer, global_step)


def main(timestamp: str):
    device = torch.device(0) if torch.cuda.is_available() else torch.device("cpu")

    # Load config
    project_dir = Path(__file__).resolve().parents[1]
    conf = OmegaConf.load(project_dir / "conf/config.yaml")
    print(OmegaConf.to_yaml(conf))

    # seed everything
    random.seed(conf.main.seed)
    torch.manual_seed(conf.main.seed)
    np.random.seed(conf.main.seed)
    torch.cuda.manual_seed_all(conf.main.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    mlflow.set_tracking_uri("https://mlflow.par.prod.crto.in/")
    mlflow.set_experiment("al.thomas_data_2_text")
    # todo: batch updates to speed up training
    #  https://www.mlflow.org/docs/latest/python_api/mlflow.tracking.html#mlflow.tracking.MlflowClient.log_batch
    global run_name
    run_name = f"{timestamp}-{conf.main.mode}"
    global tb_writer
    tb_writer = SummaryWriter(log_dir=str(project_dir / f"models/{run_name}"))
    print(f"run_name: {run_name}\n")

    with mlflow.start_run(run_name=run_name):
        for k1 in conf.keys():
            for k2 in conf[k1].keys():
                mlflow.log_param(f"{k1}_{k2}", conf[k1][k2])
        # save source files and vocab
        for f in (project_dir / "src").rglob("*.py"):
            mlflow.log_artifact(str(f), f"code/{f}")

        # load data
        dataset_train, dataset_val, dataset_test, vocab, collate_fn = prepare_data(
            data_dir=project_dir / "data", device=device
        )
        batch_size = conf.g2t.batch_size
        assert batch_size == conf.t2g.batch_size
        dataloader_train = DataLoader(
            dataset_train,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )
        dataloader_val = DataLoader(
            dataset_val,
            batch_size=batch_size,
            collate_fn=collate_fn,
        )
        dataloader_test = DataLoader(
            dataset_test,
            batch_size=batch_size,
            collate_fn=collate_fn,
        )

        # train
        train(
            conf=conf,
            device=device,
            dataloader_train=dataloader_train,
            dataloader_val=dataloader_val,
            dataloader_test=dataloader_test,
            vocab=vocab,
        )


if __name__ == "__main__":
    # filter out some hdfs warnings (from 3rd party python libraries)
    sys.stdout = WarningsFilter(sys.stdout)
    sys.stderr = WarningsFilter(sys.stderr)
    timestamp = datetime.datetime.today().strftime("%m%d%H%M%S")
    main(timestamp=timestamp)
