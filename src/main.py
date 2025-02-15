import datetime
import logging
import sys
from pathlib import Path

import mlflow
from accelerate import Accelerator, DistributedDataParallelKwargs
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer

from src.data.datasets import WebNLG2020
from src.data.formatting import (
    GraphFormat,
    GENERATE_TEXT_TOKEN,
    GENERATE_GRAPH_TOKEN,
    STYLE_TOKEN,
)
from src.eval.evaluator import EvaluatorWebNLG
from src.model import GT8FullVAE, GT8NonVAE, GT8StyleVAE
from src.trainer import Seq2seqTrainer
from src.utils import (
    WarningsFilter,
    seed_everything,
    mlflow_log_src_and_config,
    ModelSummary,
    Mode,
    CycleVAELoss,
    VAEModel,
)


def main(timestamp: str):
    # Load config
    project_dir = Path(__file__).resolve().parents[1]
    conf = OmegaConf.load(project_dir / "conf/conf_seq_to_seq.yaml")

    # multi-GPU handler
    vae_model = VAEModel(conf.vae.model)
    accelerator_kwargs = []
    if vae_model == VAEModel.style_vae:
        # for the StyleVAE, we don't use all parameters before each backward call (since we compute
        # either vae_s_x or vae_s_y), so this is necessary (https://pytorch.org/docs/stable/notes/ddp.html)
        accelerator_kwargs = [
            DistributedDataParallelKwargs(find_unused_parameters=True)
        ]
    accelerator = Accelerator(kwargs_handlers=accelerator_kwargs)

    # format logging
    logging.basicConfig(
        format="%(process)d %(asctime)s %(message)s",
        datefmt="[%H:%M:%S]",
        level=logging.INFO if accelerator.is_local_main_process else logging.ERROR,
    )

    # complete and print conf, with a specific run name
    use_loggers = accelerator.is_local_main_process and not conf.fast_dev_run
    conf.use_fp16 = accelerator.use_fp16
    conf.num_processes = accelerator.num_processes
    logging.info(OmegaConf.to_yaml(conf))
    run_name = f"{timestamp}-{conf.mode}-{conf.model}-{conf.vae.model}"
    if conf.mode == "both_unsup":
        run_name += f"-{conf.generate_method}"
    if vae_model == VAEModel.full_vae:
        run_name += f"-{conf.vae.cycle_loss}"

    # seed everything
    seed_everything(conf.seed)

    logging.info(f"run_name: {run_name}\n")
    if use_loggers:
        tb_writer = SummaryWriter(log_dir=str(project_dir / f"models/{run_name}"))
    else:
        tb_writer = None

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(conf.model)
    # add our separators as new tokens
    # (regular, not special tokens like <pad> since we need them after cleaning output sentence)
    new_tokens = [
        GraphFormat.HEAD_TOKEN,
        GraphFormat.TYPE_TOKEN,
        GraphFormat.TAIL_TOKEN,
        GraphFormat.BLANK_TOKEN,
    ]
    tokenizer.add_tokens(new_tokens)
    if vae_model == VAEModel.style_vae:
        tokenizer.add_tokens(STYLE_TOKEN)
    if not conf.specify_target_with_prefix:
        # to be used as a start_token in decoder inputs
        # these ones are special tokens, since we do not want them
        # when decoding the ids to plain text (and the generate method always
        # return the decoder_start_token_id at the beginning of the output)
        tokenizer.add_tokens(
            [GENERATE_TEXT_TOKEN, GENERATE_GRAPH_TOKEN], special_tokens=True
        )
        generate_text_tok_id = tokenizer.convert_tokens_to_ids(GENERATE_TEXT_TOKEN)
        generate_graph_tok_id = tokenizer.convert_tokens_to_ids(GENERATE_GRAPH_TOKEN)
        assert (
            tokenizer.convert_ids_to_tokens(generate_text_tok_id) == GENERATE_TEXT_TOKEN
        )
        assert (
            tokenizer.convert_ids_to_tokens(generate_graph_tok_id)
            == GENERATE_GRAPH_TOKEN
        )
    else:
        generate_text_tok_id = None
        generate_graph_tok_id = None

    # load data
    data_dir = project_dir / "data"
    datasets = {
        split: WebNLG2020(
            data_dir=data_dir, split=split, tokenizer=tokenizer, accelerator=accelerator
        )
        for split in WebNLG2020.splits
    }
    train_dataset = datasets["train"]

    # prepare model (todo: put parameters in model config and load from_config?)
    if vae_model == VAEModel.full_vae:
        model = GT8FullVAE.from_pretrained(
            conf.model,
            specify_target_with_prefix=conf.specify_target_with_prefix,
            generate_text_token_id=generate_text_tok_id,
            generate_graph_token_id=generate_graph_tok_id,
            reg_loss=conf.vae.reg,
        )
    elif vae_model == VAEModel.style_vae:
        model = GT8StyleVAE.from_pretrained(
            conf.model,
            specify_target_with_prefix=conf.specify_target_with_prefix,
            generate_text_token_id=generate_text_tok_id,
            generate_graph_token_id=generate_graph_tok_id,
            reg_loss=conf.vae.reg,
            use_style_token=conf.vae.use_style_token,
        )
    elif vae_model == VAEModel.non_vae:
        model = GT8NonVAE.from_pretrained(
            conf.model,
            specify_target_with_prefix=conf.specify_target_with_prefix,
            generate_text_token_id=generate_text_tok_id,
            generate_graph_token_id=generate_graph_tok_id,
        )
    else:
        raise ValueError

    # extend embedding matrices to include new tokens
    model.resize_token_embeddings(len(tokenizer))
    summary = ModelSummary(model, mode="top")
    logging.info(f"\n{summary}")

    trainer = Seq2seqTrainer(
        model=model,
        mode=Mode(conf.mode),
        vae_model=vae_model,
        vae_cycle_loss=CycleVAELoss(conf.vae.cycle_loss),
        vae_beta=conf.vae.beta,
        beta_n_cycle=conf.vae.beta_n_cycle,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        accelerator=accelerator,
        learning_rate=conf.lr * conf.num_processes,
        lr_scheduler=conf.lr_scheduler,
        batch_size=conf.batch_size_train,
        noise_fn=conf.sample_noise_fun,
        generate_method=conf.generate_method,
        num_epochs=conf.epochs,
        tensorboard_writer=tb_writer,
        log_path=project_dir / f"models/{run_name}",
        log_every_n_steps=conf.log_every_n_steps,
        max_grad_norm=conf.max_grad_norm,
        max_training_steps=3 if conf.fast_dev_run else -1,
    )

    evaluator = EvaluatorWebNLG(
        run_name=run_name,
        mode=Mode(conf.mode),
        datasets=datasets,
        tokenizer=tokenizer,
        accelerator=accelerator,
        ddp_model=trainer.ddp_model,
        batch_size=conf.batch_size_val,
        num_beams_t2g=conf.num_beams_t2g,
        num_beams_g2t=conf.num_beams_g2t,
        log_path=project_dir / f"models/{run_name}",
        checkpoints=conf.checkpoints,
        tensorboard_writer=tb_writer,
        limit_samples=20 if conf.fast_dev_run else False,
    )
    trainer.set_evaluator(evaluator)

    if use_loggers:
        mlflow.set_tracking_uri(conf.mlflow.tracking_uri)
        mlflow.set_experiment(conf.mlflow.experiment_name)
        with mlflow.start_run(run_name=run_name):
            mlflow_log_src_and_config(conf, project_dir)
            # train model
            trainer.train()
    else:
        # don't log anything to mlflow
        trainer.train()


if __name__ == "__main__":
    # filter out some hdfs warnings (from 3rd party python libraries)
    sys.stdout = WarningsFilter(sys.stdout)
    sys.stderr = WarningsFilter(sys.stderr)
    main(timestamp=datetime.datetime.today().strftime("%m%d%H%M%S"))
