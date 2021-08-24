import logging
import random
from pathlib import Path
from typing import List

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import (
    AdamW,
    get_scheduler,
    default_data_collator,
    PreTrainedTokenizer,
)
from accelerate import Accelerator

from src.data.datasets import Seq2seqDataset
from src.data.formatting import add_prefix
from src.data.noise import existing_noise_functions
from src.eval.evaluator import EvaluatorWebNLG
from src.utils import MyLogger, Mode, AutoLoss, CycleLoss, frange_cycle_zero_linear


class Seq2seqTrainer:
    # to be set after trainer init (we need to create Trainer with accelerator first)
    evaluator: EvaluatorWebNLG

    def __init__(
        self,
        model,
        mode: Mode,
        cycle_loss: str,
        auto_loss: str,
        vae_beta: float,
        beta_n_cycle: int,
        tokenizer: PreTrainedTokenizer,
        train_dataset: Seq2seqDataset,
        accelerator: Accelerator,
        learning_rate: float,
        lr_scheduler: str,
        batch_size: int,
        num_epochs: int,
        noise_fn: List[str],
        generate_method: str,
        tensorboard_writer: SummaryWriter,
        log_path: Path,
        log_every_n_steps: int,
        max_grad_norm: float,
        max_training_steps: int = -1,
    ):
        self.mode = mode
        self.cycle_loss = cycle_loss
        self.auto_loss = auto_loss
        self.tokenizer = tokenizer

        # training
        optimizer = AdamW(model.parameters(), lr=learning_rate)
        train_dataloader = DataLoader(
            train_dataset,
            shuffle=True,
            batch_size=batch_size,
            collate_fn=default_data_collator,
        )
        # prepare model and data for multi gpu training (if necessary)
        self.accelerator = accelerator
        (
            self.ddp_model,
            self.optimizer,
            self.train_dataloader,
        ) = accelerator.prepare(model, optimizer, train_dataloader)

        # training parameters
        if max_training_steps > 0:
            self.num_training_steps = max_training_steps * num_epochs
        else:
            self.num_training_steps = num_epochs * len(self.train_dataloader)
        self.num_epochs = num_epochs
        self.lr_scheduler = get_scheduler(
            name=lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=0,
            num_training_steps=self.num_training_steps,
        )
        self.batch_size = batch_size
        self.max_seq_length = train_dataset.max_seq_length
        self.max_grad_norm = max_grad_norm
        self.noise_functions = noise_fn
        self.generate_method = generate_method
        self.use_cyclical_beta_schedule = beta_n_cycle > -1
        if self.use_cyclical_beta_schedule:
            self.betas = frange_cycle_zero_linear(self.num_training_steps, beta_n_cycle)
        else:
            self.beta = vae_beta

        # logging
        self.log_path = log_path
        self.log_every_n_steps = log_every_n_steps
        self.logger = MyLogger(
            tensorboard_writer=tensorboard_writer,
            log_every_n_steps=log_every_n_steps,
            accelerator=accelerator,
            use_loggers=tensorboard_writer is not None,
        )

    def set_evaluator(self, evaluator: EvaluatorWebNLG):
        self.evaluator = evaluator

    def predict(self, input_ids: torch.Tensor, target: str):
        model = self.accelerator.unwrap_model(self.ddp_model)
        prediction_ids = model.generate_with_target(
            input_ids=input_ids,
            target=target,
            tokenizer=self.tokenizer,
            max_seq_length=self.max_seq_length,
            method=self.generate_method,
        )
        # multi-GPU: no need to gather predictions across processes yet, since the
        # predictions are to be used in training (gathering is down after the loss is computed)
        return prediction_ids

    def teach_model_one_step(
        self, input_ids: torch.Tensor, label_ids: torch.Tensor, target: str, beta: float
    ):
        """

        Args:
            input_ids: input sequence (text/graph tokenized batch, already on device)
            label_ids: label (ground truth graph/text as a tokenized sequence)
            target: 'text' or 'graph', depending on the format of label sequences. Will
                determine the prefix to add to input_ids, or the token to specify to the decoder

        Returns:
            loss

        """
        model = self.accelerator.unwrap_model(self.ddp_model)
        if model.specify_target_with_prefix:
            # add the prefix "generate graph/text" to the input
            # (if model.specify_target_with_prefix=False, the model handles this itself
            # and uses the right decoder_start_token_id in the forward)
            input_ids = add_prefix(
                input_ids=input_ids,
                target=target,
                tokenizer=self.tokenizer,
                max_seq_len=self.max_seq_length,
            )
        att_mask_input = self.get_att_mask(input_ids)

        # todo: check if we need to set pad token ids to -100 in the labels,
        #  to ignore them when computing the loss
        self.ddp_model.train()
        outputs = self.ddp_model(
            input_ids=input_ids,
            attention_mask=att_mask_input,
            labels=label_ids,
            target=target,
            beta=beta,
        )
        loss = outputs.loss
        self.accelerator.backward(loss)
        return outputs

    def train(self):
        global_step = 0
        logging.info("Training...")
        logging.info(f"     num_epochs: {self.num_epochs}")

        for epoch in range(self.num_epochs):
            for batch in tqdm(
                self.train_dataloader,
                desc=f"[ep{epoch}]",
                disable=not self.accelerator.is_local_main_process,
            ):
                # stop training if a max number of steps was specified
                if global_step > self.num_training_steps * (epoch + 1):
                    break

                # get batch data
                text_ids = batch["text_ids"]
                graph_ids = batch["graph_ids"]
                att_mask_text = batch["att_mask_text"]
                att_mask_graph = batch["att_mask_graph"]
                # simply make sure our get_att_mask method works
                assert (att_mask_graph == self.get_att_mask(graph_ids)).all()
                assert (att_mask_text == self.get_att_mask(text_ids)).all()

                # training step
                t2g_outputs = None
                g2t_outputs = None
                text_outputs = None
                graph_outputs = None
                syn_text_ids = None  # synthetic text and graphs
                syn_graph_ids = None
                noisy_text_ids = None
                noisy_graph_ids = None

                # select beta
                if self.use_cyclical_beta_schedule:
                    beta_t = self.betas[global_step]
                else:
                    beta_t = self.beta

                if self.mode == Mode.t2g:
                    t2g_outputs = self.teach_model_one_step(
                        input_ids=text_ids,
                        label_ids=graph_ids,
                        target="graph",
                        beta=beta_t,
                    )
                elif self.mode == Mode.g2t:
                    g2t_outputs = self.teach_model_one_step(
                        input_ids=graph_ids,
                        label_ids=text_ids,
                        target="text",
                        beta=beta_t,
                    )
                elif self.mode == Mode.both_sup:
                    g2t_outputs = self.teach_model_one_step(
                        input_ids=graph_ids,
                        label_ids=text_ids,
                        target="text",
                        beta=beta_t,
                    )
                    t2g_outputs = self.teach_model_one_step(
                        input_ids=text_ids,
                        label_ids=graph_ids,
                        target="graph",
                        beta=beta_t,
                    )
                elif self.mode == Mode.both_unsup:
                    # todo: make sure the predictions are correctly formatted, especially the attention mask
                    #   -> does it start with an unnecessary padding token?
                    #   -> should we hide the prefix ("Generate graph/text:") to the encoder in the input?

                    # todo: samples will be correlated with denoising autoencoding+backtranslation
                    #   -> do one epoch of each like GT-BT?
                    #   -> or see what Lample does?
                    if self.auto_loss == AutoLoss.denoising:
                        # text denoising auto-encoder step
                        noisy_text_ids = self.get_noisy_inputs(text_ids, is_graph=False)
                        text_outputs = self.teach_model_one_step(
                            input_ids=noisy_text_ids,
                            label_ids=text_ids,
                            target="text",
                            beta=beta_t,
                        )
                        # graph denoising auto-encoder step
                        noisy_graph_ids = self.get_noisy_inputs(
                            graph_ids, is_graph=False
                        )
                        graph_outputs = self.teach_model_one_step(
                            input_ids=noisy_graph_ids,
                            label_ids=graph_ids,
                            target="graph",
                            beta=beta_t,
                        )
                    elif self.auto_loss == AutoLoss.vae:
                        text_outputs = self.teach_model_one_step(
                            input_ids=text_ids,
                            label_ids=text_ids,
                            target="text",
                            beta=beta_t,
                        )
                        graph_outputs = self.teach_model_one_step(
                            input_ids=graph_ids,
                            label_ids=graph_ids,
                            target="graph",
                            beta=beta_t,
                        )
                    else:
                        raise ValueError

                    if (
                        self.cycle_loss == CycleLoss.regular
                        or self.cycle_loss == CycleLoss.vae
                    ):
                        # g2t unsupervised step
                        syn_graph_ids = self.predict(input_ids=text_ids, target="graph")
                        g2t_outputs = self.teach_model_one_step(
                            input_ids=syn_graph_ids,
                            label_ids=text_ids,
                            target="text",
                            beta=beta_t,
                        )
                        # t2g unsupervised step
                        syn_text_ids = self.predict(input_ids=graph_ids, target="text")
                        t2g_outputs = self.teach_model_one_step(
                            input_ids=syn_text_ids,
                            label_ids=graph_ids,
                            target="graph",
                            beta=beta_t,
                        )
                    else:
                        raise ValueError
                else:
                    raise ValueError

                self.accelerator.clip_grad_norm_(
                    self.ddp_model.parameters(), self.max_grad_norm
                )
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
                global_step += 1

                # log training info (metrics and text sequences)
                self.log_metrics(
                    global_step,
                    epoch,
                    t2g_outputs=t2g_outputs,
                    g2t_outputs=g2t_outputs,
                    text_outputs=text_outputs,
                    graph_outputs=graph_outputs,
                )
                self.log_training_samples(
                    global_step,
                    epoch,
                    text_ids=text_ids,
                    graph_ids=graph_ids,
                    syn_text_ids=syn_text_ids,
                    syn_graph_ids=syn_graph_ids,
                    noisy_text_ids=noisy_text_ids,
                    noisy_graph_ids=noisy_graph_ids,
                    t2g_logits=t2g_outputs.logits if t2g_outputs else None,
                    g2t_logits=g2t_outputs.logits if g2t_outputs else None,
                )

            # free GPU memory before eval
            t2g_outputs = None
            g2t_outputs = None
            text_outputs = None
            graph_outputs = None
            # evaluate after each epoch (and save model checkpoint if necessary)
            self.evaluator.on_epoch_end(epoch)
            self.logger.send_current_logs()

        # evaluate on test set
        #   todo: remove when tuning hyperparameters, to make sure we don't overfit on test set
        self.evaluator.on_training_end()

    @staticmethod
    def get_att_mask(input_ids: torch.Tensor):
        # attention mask: 0 if it's a padding token, 1 otherwise
        # also type as input ids (tensor of integers)
        att_mask = (input_ids != 0).type_as(input_ids)
        return att_mask

    def get_noisy_inputs(
        self,
        input_ids: torch.Tensor,
        is_graph: bool,
    ):
        # decode input ids
        texts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        # add noise to the texts/graphs
        noisy_inputs = []
        for text in texts:
            noise_fun_name = random.choice(self.noise_functions)
            noise_fun = existing_noise_functions[noise_fun_name]
            # todo: try composing noise functions?
            noisy_text, _ = noise_fun(text, is_graph=is_graph)
            noisy_inputs.append(noisy_text)
        # tokenize back
        batch_encoding = self.tokenizer(
            noisy_inputs,
            max_length=self.max_seq_length,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )
        batch_encoding = batch_encoding.to(input_ids.device)
        noisy_ids = batch_encoding.input_ids
        return noisy_ids

    def log_metrics(self, global_step: int, epoch: int, **kwargs):
        metrics = {
            "train/learning_rate": self.lr_scheduler.get_last_lr()[0],
            "train/epoch": epoch,
        }
        for m in ["t2g", "g2t", "text", "graph"]:
            # for each mode, log our regular and vae metrics
            if kwargs.get(f"{m}_outputs", None) is not None:
                outputs = kwargs[f"{m}_outputs"]
                metrics[f"train/loss_{m}"] = outputs.loss.item()
                if "recon_loss" in outputs:
                    metrics[f"train/recon_loss_{m}"] = outputs.recon_loss.item()
                if "reg_loss" in outputs:
                    metrics[f"train/reg_loss_{m}"] = outputs.reg_loss.item()

        self.logger.log_metrics(metrics, step=global_step)

    def log_training_samples(self, global_step: int, epoch: int, **kwargs):
        if (
            global_step % self.log_every_n_steps == 0
            and self.accelerator.is_local_main_process
        ):
            # make missing predictions (e.g. in supervised mode, where we don't
            # need to generate fake samples)
            if kwargs["syn_graph_ids"] is None:
                kwargs["syn_graph_ids"] = self.predict(
                    input_ids=kwargs["text_ids"], target="graph"
                )
            if kwargs["syn_text_ids"] is None:
                kwargs["syn_text_ids"] = self.predict(
                    input_ids=kwargs["graph_ids"], target="text"
                )
            # convert logits (obtained from the forward call with teacher forcing)
            # into token id sequences
            if kwargs["t2g_logits"] is not None:
                t2g_logits = kwargs.pop("t2g_logits")
                kwargs["tf_graph_ids"] = t2g_logits.argmax(dim=-1)
            if kwargs["g2t_logits"] is not None:
                g2t_logits = kwargs.pop("g2t_logits")
                kwargs["tf_text_ids"] = g2t_logits.argmax(dim=-1)

            # in the end we want:
            # training_samples = {"noisy_text": ["sentence", "sentence2", "sentence3"]}
            training_samples = {}
            for name, token_ids in kwargs.items():
                if token_ids is None:
                    continue
                # decode the tensor of token ids into a list of strings
                # (one per example of the batch)
                sentences = self.tokenizer.batch_decode(
                    token_ids, skip_special_tokens=True
                )
                if name.endswith("_ids"):
                    name = name[:-4]
                training_samples[name] = sentences

            # format log text
            log = f"[{global_step}] {', '.join(training_samples.keys())}\n"
            batch_size = len(training_samples["text"])
            batch_logs = ["" for _ in range(batch_size)]
            for sentences in training_samples.values():
                for i, s in enumerate(sentences):
                    # concatenate all sentences for the same batch example together
                    batch_logs[i] += f"{s}\n"
            log += "-\n".join(batch_logs)

            # save logs to disk
            logs_path = self.log_path / f"training/{epoch}.txt"
            self.logger.log_text(
                text=log,
                file_path=logs_path,
                folder_name="training",
                one_time_log=False,  # don't log to mlflow until the end of the epoch
            )
