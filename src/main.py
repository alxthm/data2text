import datetime
import sys
from pathlib import Path

import mlflow
import torch
import torch.optim
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.model.cycle_cvae import CycleCVAE
from src.utils import (
    WarningsFilter,
    seed_everything,
    mlflow_log_src_and_config,
    ModelSummary,
)


def main(timestamp: str):
    # Load config
    project_dir = Path(__file__).resolve().parents[1]
    conf = OmegaConf.load(project_dir / "conf/config.yaml")
    print(OmegaConf.to_yaml(conf))

    # seed everything
    device = torch.device(0) if torch.cuda.is_available() else torch.device("cpu")
    seed_everything(conf.seed)

    mlflow.set_tracking_uri("https://mlflow.par.prod.crto.in/")
    mlflow.set_experiment("al.thomas_data_2_text")
    run_name = f"{timestamp}-{'sup' if conf.supervised else 'unsup'}"
    tb_writer = SummaryWriter(log_dir=str(project_dir / f"models/{run_name}"))
    print(f"run_name: {run_name}\n")

    with mlflow.start_run(run_name=run_name):
        mlflow_log_src_and_config(conf, project_dir)

        # load data
        if conf.dataset == "genwiki":
            from src.data.genwiki import prepare_data
        elif conf.dataset == "webnlg":
            from src.data.webnlg import prepare_data
        else:
            raise ValueError

        (
            dataset_train_t2g,
            dataset_train_g2t,
            dataset_val,
            dataset_test,
            vocab,
            collate_fn_train,
            collate_fn_eval,
        ) = prepare_data(
            data_dir=project_dir / "data", device=device, is_supervised=conf.supervised
        )
        dataloader_train_t2g = DataLoader(
            dataset_train_t2g,
            batch_size=conf.batch_size,
            shuffle=True,
            collate_fn=collate_fn_train,
        )
        dataloader_train_g2t = DataLoader(
            dataset_train_g2t,
            batch_size=conf.batch_size,
            shuffle=True,
            collate_fn=collate_fn_train,
        )
        dataloader_val = DataLoader(
            dataset_val,
            batch_size=conf.batch_size,
            collate_fn=collate_fn_eval,
        )
        dataloader_test = DataLoader(
            dataset_test,
            batch_size=conf.batch_size,
            collate_fn=collate_fn_eval,
        )

        # prepare model
        model = CycleCVAE(
            text_vocab=vocab["text"],
            ent_vocab=vocab["entity"],
            rel_vocab=vocab["relation"],
            tokenizer=collate_fn_eval.tokenizer,
            tot_epoch=conf.epoch,
            dim_h=conf.dim_h,
            dim_z=conf.g2t.dim_z,
            enc_lstm_layers=conf.g2t.enc_lstm_layers,
            n_head=conf.g2t.n_head,
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
            gradient_clip_val=conf.grad_clip,
            run_name=run_name,
            device=device,
            is_supervised=conf.supervised,
        )
        model.to(device)
        summary = ModelSummary(model, mode="top")
        print(summary)

        # train
        global_step = 0
        for ep in range(0, conf.epoch):
            # training epoch
            for batch_t2g, batch_g2t in tqdm(
                zip(dataloader_train_t2g, dataloader_train_g2t), desc=f"[train][ep{ep}]"
            ):
                metrics = model.training_step(
                    batch_t2g, batch_g2t, global_step=global_step
                )

                # log metrics
                if global_step % conf.log_every_n_steps == 0:
                    mlflow.log_metrics(metrics, step=global_step)
                    for k, v in metrics.items():
                        tb_writer.add_scalar(k, v, global_step=global_step)
                    if conf.log_gradients:
                        model.log_gradients(tb_writer, global_step, print_warning=True)
                global_step += 1

            # validation epoch
            model.eval()
            with torch.no_grad():
                model.on_eval_epoch_start()
                for batch in tqdm(dataloader_val, desc=f"[val][ep{ep}]"):
                    model.eval_step(batch)
                model.on_eval_epoch_end("val", tb_writer, global_step)

                if ep % 15 == 0 and conf.save_checkpoints:
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


if __name__ == "__main__":
    # filter out some hdfs warnings (from 3rd party python libraries)
    sys.stdout = WarningsFilter(sys.stdout)
    sys.stderr = WarningsFilter(sys.stderr)
    timestamp = datetime.datetime.today().strftime("%m%d%H%M%S")
    main(timestamp=timestamp)
