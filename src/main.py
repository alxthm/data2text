import datetime
import datetime
import random
import sys
from collections import defaultdict
from pathlib import Path

import mlflow
import numpy as np
import torch
import torch.optim
from omegaconf import OmegaConf
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.data.webnlg import write_txt, prepare_data, tokenizer
from src.model.cycle_cvae import CycleCVAE
from src.model.t2g import write_t2g_log
from src.utils import WarningsFilter

bleu = Bleu(4)
meteor = Meteor()
rouge = Rouge()
cider = Cider()
tb_writer: SummaryWriter = None
run_name = ""

# def prep_model(conf, vocab):
#     g2t_model = GraphWriter(copy.deepcopy(conf.g2t), vocab)
#     t2g_model = T2G(
#         relation_types=len(vocab["relation"]),
#         d_model=conf.t2g.nhid,
#         dropout=conf.t2g.drop,
#     )
#     return g2t_model, t2g_model


# def train_g2t_one_step(batch, model, optimizer, conf, global_step):
#     model.train()
#     optimizer.zero_grad()
#     pred, pred_c, kl_div = model(batch)
#     recon_loss = F.nll_loss(
#         pred.reshape(-1, pred.shape[-1]), batch["g2t_tgt"].reshape(-1), ignore_index=0
#     )
#     kl_anneal = min(1.0, (global_step + 100) / (global_step + 10000))
#     loss = recon_loss + kl_anneal * 8.0 / 385 * kl_div
#     loss.backward()
#     nn.utils.clip_grad_norm_(model.parameters(), conf.clip)
#     optimizer.step()
#     return loss.item(), recon_loss.item(), kl_div.item()
#
#
# def train_t2g_one_step(batch, model, optimizer, conf, device, t2g_weight=None):
#     model.train()
#     if t2g_weight is not None:
#         # category weights
#         t2g_weight = torch.from_numpy(t2g_weight).float().to(device)
#     optimizer.zero_grad()
#     # logits for the relation type, between every entity tuple of the sentence
#     pred = model(batch)  # (bs, ne, ne, num_relations)
#     loss = F.nll_loss(
#         pred.contiguous().view(-1, pred.shape[-1]),
#         batch["t2g_tgt"].contiguous().view(-1),  # initially shape (bs, ne, ne)
#         ignore_index=0,
#         weight=t2g_weight,
#     )
#     loss.backward()
#     nn.utils.clip_grad_norm_(model.parameters(), conf.clip)
#     optimizer.step()
#     return loss.item()
#
#
# def supervise(
#     batch,
#     model_g2t,
#     model_t2g,
#     optimizerG2T,
#     optimizerT2G,
#     conf,
#     t2g_weight,
#     device,
#     global_step,
# ):
#     model_g2t.blind, model_t2g.blind = False, False
#     loss_t2g = train_t2g_one_step(
#         batch, model_t2g, optimizerT2G, conf.t2g, device, t2g_weight=t2g_weight
#     )
#     loss_g2t, recon_loss, kl_div = train_g2t_one_step(
#         batch, model_g2t, optimizerG2T, conf.g2t, global_step
#     )
#     return loss_t2g, loss_g2t, recon_loss, kl_div


def evaluate(dataloader_val, t2g_model, g2t_model, vocab, global_step, beam_size):
    ent_vocab = vocab["entity"]
    rel_vocab = vocab["relation"]
    text_vocab = vocab["text"]
    wf_t2g = open(f"/tmp/{run_name}_t2g_out.txt", "w", encoding="utf-8")
    t2g_hyp, t2g_ref, t2g_pos_label = [], [], []
    g2t_hyp = []
    g2t_ref = []
    g2t_same = []
    # todo: use pytorch metrics?

    for i, batch in tqdm(enumerate(dataloader_val)):
        batch_size = len(batch["original_ent_text"])

        # t2g
        pred = t2g_model(batch)
        _pred = pred.view(-1, pred.shape[-1]).argmax(-1).cpu().long().tolist()
        _gold = batch["t2g_tgt"].view(-1).cpu().long().tolist()
        tpred = pred.argmax(-1).cpu().numpy()
        tgold = batch["t2g_tgt"].cpu().numpy()
        # save predictions as plain text
        for j in range(batch_size):
            ents = [
                [y for y in ent_vocab(x) if y[0] != "<"]
                for x in batch["original_ent_text"][j]
            ]
            wf_t2g.write("=====================\n")
            wf_t2g.write("--- Predictions\n")
            write_t2g_log(wf_t2g, rel_vocab, tpred[j], ents)
            wf_t2g.write("--- Target\n")
            write_t2g_log(wf_t2g, rel_vocab, tgold[j], ents)
        # compute f1 metrics
        pred, gold = [], []
        for j in range(len(_gold)):
            if (
                _gold[j] > 0
            ):  # ignore <PAD> -> todo: simply use ignore_index of F1 metric?
                pred.append(_pred[j])
                gold.append(_gold[j])
        t2g_pos_label.extend([x for x in gold if x != 3])  # 3 is no relation
        t2g_hyp.extend(pred)
        t2g_ref.extend(gold)

        # g2t
        seq = g2t_model(batch, beam_size=beam_size)  # (bs, max_sent_len)
        r = write_txt(batch, batch["g2t_tgt"], text_vocab)
        h = write_txt(batch, seq, text_vocab)
        g2t_same.extend(
            [
                str(batch["original_raw_relation"][i])
                + str(batch["original_ent_text"][i])
                for i in range(batch_size)
            ]
        )
        # save text predictions for evaluation and logging at the end of val epoch
        g2t_hyp.extend(h)
        g2t_ref.extend(r)

    # compute metrics over all validation batch and log results
    wf_t2g.close()
    pos_label = list(set(t2g_pos_label))
    f1_micro = f1_score(
        t2g_ref, t2g_hyp, average="micro", labels=pos_label, zero_division=0
    )
    f1_macro = f1_score(
        t2g_ref, t2g_hyp, average="macro", labels=pos_label, zero_division=0
    )
    mlflow.log_artifact(f"/tmp/{run_name}_t2g_out.txt", f"t2g_out/{global_step}")
    mlflow.log_metrics(
        {"dev_f1_micro": f1_micro, "dev_f1_macro": f1_macro}, step=global_step
    )
    tb_writer.add_scalar("dev_f1_micro", f1_micro, global_step=global_step)
    tb_writer.add_scalar("dev_f1_macro", f1_macro, global_step=global_step)

    # g2t - format predictions into text
    unq_hyp = {}
    unq_ref = defaultdict(list)
    g2t_hyp = [x[0] for x in g2t_hyp]
    g2t_ref = [x[0] for x in g2t_ref]
    idxs, g2t_same = list(zip(*sorted(enumerate(g2t_same), key=lambda x: x[1])))
    ptr = 0
    for i in range(len(g2t_hyp)):
        if i > 0 and g2t_same[i] != g2t_same[i - 1]:
            ptr += 1
        unq_hyp[ptr] = g2t_hyp[idxs[i]]
        unq_ref[ptr].append(g2t_ref[idxs[i]])
    unq_hyp = sorted(unq_hyp.items(), key=lambda x: x[0])
    unq_ref = sorted(unq_ref.items(), key=lambda x: x[0])
    g2t_hyp = [x[1] for x in unq_hyp]
    g2t_ref = [[x.lower() for x in y[1]] for y in unq_ref]

    # log predictions
    assert len(unq_ref) == len(unq_hyp)
    with open(f"/tmp/{run_name}_g2t_out.txt", "w", encoding="utf-8") as wf_g2t:
        for i in range(len(g2t_hyp)):
            wf_g2t.write(
                f"{i}\n" f"pred: {str(g2t_hyp[i])}\n" f"tgt: {str(g2t_ref[i])}\n"
            )
    mlflow.log_artifact(f"/tmp/{run_name}_g2t_out.txt", f"g2t_out/{global_step}")
    # compute NLG metrics
    g2t_hyp = dict(zip(range(len(g2t_hyp)), [[x.lower()] for x in g2t_hyp]))
    g2t_ref = dict(zip(range(len(g2t_ref)), g2t_ref))
    bleu_score = bleu.compute_score(g2t_ref, g2t_hyp)
    metrics = {
        "BLEU INP": len(g2t_hyp),
        "BLEU 1": bleu_score[0][0],
        "BLEU 2": bleu_score[0][1],
        "BLEU 3": bleu_score[0][2],
        "BLEU 4": bleu_score[0][3],
        "METEOR": meteor.compute_score(g2t_ref, g2t_hyp)[0],
        "ROUGE_L": (rouge.compute_score(g2t_ref, g2t_hyp)[0]),
        "Cider": cider.compute_score(g2t_ref, g2t_hyp)[0],
    }
    mlflow.log_metrics(metrics)
    for k, v in metrics.items():
        tb_writer.add_scalar(k, v, global_step=global_step)


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
    )
    model.to(device)

    #
    # optimizerG2T = torch.optim.Adam(
    #     model_g2t.parameters(),
    #     lr=conf.g2t.lr,
    #     weight_decay=conf.g2t.weight_decay,
    # )
    # schedulerG2T = get_cosine_schedule_with_warmup(
    #     optimizer=optimizerG2T,
    #     num_warmup_steps=400,
    #     num_training_steps=800 * conf.main.epoch,
    # )
    # optimizerT2G = torch.optim.Adam(
    #     model_t2g.parameters(),
    #     lr=conf.t2g.lr,
    #     weight_decay=conf.t2g.weight_decay,
    # )
    # schedulerT2G = get_cosine_schedule_with_warmup(
    #     optimizer=optimizerT2G,
    #     num_warmup_steps=400,
    #     num_training_steps=800 * conf.main.epoch,
    # )

    # t2g_weight = [vocab["relation"].wf.get(x, 0) for x in vocab["relation"].i2s]
    # t2g_weight[0] = 0
    # max_w = max(t2g_weight)
    # t2g_weight = np.array(t2g_weight).astype("float32")
    # t2g_weight = (max_w + 1000) / (t2g_weight + 1000)

    global_step = 0
    for i in range(0, conf.main.epoch):
        # training epoch

        # model_t2g.train()
        # model_g2t.train()
        for j, batch in enumerate(tqdm(dataloader_train, desc=f"ep{i}")):
            metrics = model.training_step(batch, global_step=global_step)
            model.on_train_batch_end(tb_writer, global_step)
            mlflow.log_metrics(metrics, step=global_step)
            for k, v in metrics.items():
                tb_writer.add_scalar(k, v, global_step=global_step)
            global_step += 1

        # validation epoch
        model.g2t_model.blind, model.t2g_model.blind = False, False
        model.t2g_model.eval()
        model.g2t_model.eval()
        with torch.no_grad():
            evaluate(
                dataloader_val,
                model.t2g_model,
                model.g2t_model,
                vocab,
                global_step,
                conf.g2t.beam_size,
            )
            if i % 15 == 0:
                # save model checkpoints
                t2g_save = f"{conf.t2g.save}_ep{i}"
                torch.save(model.t2g_model.state_dict(), t2g_save)
                mlflow.log_artifact(t2g_save, t2g_save)
                g2t_save = f"{conf.g2t.save}_ep{i}"
                torch.save(model.g2t_model.state_dict(), g2t_save)
                mlflow.log_artifact(g2t_save, g2t_save)

    # test
    torch.save(model.g2t_model.state_dict(), f"{conf.g2t.save}_final")
    torch.save(model.t2g_model.state_dict(), f"{conf.t2g.save}_final")
    evaluate(
        dataloader_test,
        model.t2g_model,
        model.g2t_model,
        vocab,
        global_step,
        conf.g2t.beam_size,
    )


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
