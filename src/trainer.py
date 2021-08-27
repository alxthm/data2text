import logging
import random
from pathlib import Path
from typing import List, Optional

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
from src.data.formatting import add_target_prefix, add_style_prefix
from src.data.noise import existing_noise_functions
from src.eval.evaluator import EvaluatorWebNLG
from src.model import (
    GT8ModelOutput,
    GT8FullVAE,
    VariationalT5EncoderOutput,
    GT8StyleVAE,
)
from src.utils import MyLogger, Mode, frange_cycle_zero_linear, CycleVAELoss, VAEModel


class Seq2seqTrainer:
    # to be set after trainer init (we need to create Trainer with accelerator first)
    evaluator: EvaluatorWebNLG

    def __init__(
        self,
        model,
        mode: Mode,
        vae_cycle_loss: CycleVAELoss,
        vae_model: VAEModel,
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
        self.ddp_model, optimizer, train_dataloader = accelerator.prepare(
            model, optimizer, train_dataloader
        )
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader

        # training parameters
        self.batch_size = batch_size
        # max training steps per epoch
        if max_training_steps > 0:
            # stop early for testing purposes
            self.max_training_steps = max_training_steps
        else:
            self.max_training_steps = len(self.train_dataloader)
        num_training_steps = num_epochs * self.max_training_steps
        self.num_epochs = num_epochs
        self.lr_scheduler = get_scheduler(
            name=lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps,
        )
        self.max_seq_length = train_dataset.max_seq_length
        self.max_grad_norm = max_grad_norm
        self.noise_functions = noise_fn
        self.generate_method = generate_method
        # VAE specific code
        self.vae_model = vae_model
        self.use_vae = (
            vae_model == VAEModel.full_vae
            or vae_model == VAEModel.style_vae
            or vae_model == VAEModel.added_style_vae
        )
        if self.use_vae:
            self.vae_cycle_loss = vae_cycle_loss
            self.use_cyclical_beta_schedule = beta_n_cycle > -1
            if self.use_cyclical_beta_schedule:
                self.betas = frange_cycle_zero_linear(num_training_steps, beta_n_cycle)
            else:
                # constant beta coefficient -> specify once to the model
                model.beta = vae_beta

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
        self,
        input_ids: Optional[torch.Tensor],
        label_ids: torch.Tensor,
        target: str,
        retain_graph: bool = False,
        source: str = None,
    ) -> GT8ModelOutput:

        """
        Run a forward pass in the model using input_ids, and backward the loss wrt label_ids.
        If necessary, append prefix to input_ids (to specify text/graph target).

        Args:
            input_ids: input sequence (text/graph tokenized batch, already on device).
                Should be None if encoder_outputs is given.
            label_ids: label (ground truth graph/text as a tokenized sequence)
            target: 'text' or 'graph', depending on the format of label sequences. Will
                determine the prefix to add to input_ids, or the token to specify to the decoder
            retain_graph: passed to the backward (default: False)
            source: None by default, otherwise 'text' or 'graph' and passed to model forward

        Returns:
            model_outputs

        """
        model = self.accelerator.unwrap_model(self.ddp_model)
        if model.specify_target_with_prefix:
            # add the prefix "generate graph/text" to the input
            # (if model.specify_target_with_prefix=False, the model handles this itself
            # and uses the right decoder_start_token_id in the forward)
            input_ids = add_target_prefix(
                input_ids=input_ids,
                target=target,
                tokenizer=self.tokenizer,
                max_seq_len=self.max_seq_length,
            )

        # if necessary (for the style_vae), add a special [STYLE] token and specify the source format
        kwargs = {}
        if (
            self.vae_model == VAEModel.style_vae
            or self.vae_model == VAEModel.added_style_vae
        ):
            kwargs["source"] = source
        if self.vae_model == VAEModel.style_vae:
            # todo: careful when merging to check that it's the styleVAE with prefix
            input_ids = add_style_prefix(input_ids=input_ids, tokenizer=self.tokenizer)

        att_mask_input = self.get_att_mask(input_ids)
        encoder_outputs = None

        self.ddp_model.train()
        outputs = self.ddp_model(
            input_ids=input_ids,
            attention_mask=att_mask_input,
            labels=label_ids,
            target=target,
            encoder_outputs=encoder_outputs,
            **kwargs,
        )
        # we call loss.backward() here to free GPU memory for the next steps
        # -> computed gradients are kept in the leaf variables (the parameters)
        # -> but the computational graph is removed elsewhere
        # -> equivalent to calling backward on the sum of the losses, since gradients
        #   are added until we call .zero_grad()
        self.accelerator.backward(outputs.loss, retain_graph=retain_graph)
        return outputs

    def teach_model_one_step_with_vae_z(
        self,
        label_ids: torch.Tensor,
        target: str,
        vae_z: Optional[torch.FloatTensor] = None,
    ):
        """
        (Only used by the GT8FullVAE model, with dual cycle loss)
        Run a forward pass re-using some vae_z already computed from some input_ids,
        instead of input_ids
        """
        # instead of using input ids, we use
        encoder_outputs = VariationalT5EncoderOutput(vae_z=vae_z)
        self.ddp_model.train()
        outputs = self.ddp_model(
            labels=label_ids,
            target=target,
            encoder_outputs=encoder_outputs,
        )
        self.accelerator.backward(outputs.loss)
        return outputs

    def compute_loss_unsup_non_vae(
        self, text_ids: torch.Tensor, graph_ids: torch.Tensor
    ):
        # -- auto loss (denoising auto-encoding)
        noisy_text_ids = self.get_noisy_inputs(text_ids, is_graph=False)
        noisy_graph_ids = self.get_noisy_inputs(graph_ids, is_graph=True)
        text_outputs = self.teach_model_one_step(
            noisy_text_ids, text_ids, target="text"
        )
        graph_outputs = self.teach_model_one_step(
            noisy_graph_ids, graph_ids, target="graph"
        )

        # -- cycle loss
        syn_graph_ids = self.predict(input_ids=text_ids, target="graph")
        syn_text_ids = self.predict(input_ids=graph_ids, target="text")
        g2t_outputs = self.teach_model_one_step(syn_graph_ids, text_ids, target="text")
        t2g_outputs = self.teach_model_one_step(syn_text_ids, graph_ids, target="graph")

        for out in [text_outputs, graph_outputs, g2t_outputs, t2g_outputs]:
            # since we have a non VAE model, reg_loss is None and loss=recon_loss
            assert out.reg_loss is None
        return {
            "text": text_outputs,
            "graph": graph_outputs,
            "t2g": t2g_outputs,
            "g2t": g2t_outputs,
        }

    def compute_loss_unsup_vae_single(
        self, text_ids: torch.Tensor, graph_ids: torch.Tensor
    ):
        # -- auto loss (regular VAE)
        text_outputs = self.teach_model_one_step(
            text_ids, text_ids, target="text", source="text"
        )
        graph_outputs = self.teach_model_one_step(
            graph_ids, graph_ids, target="graph", source="graph"
        )

        # -- cycle loss (single)
        syn_graph_ids = self.predict(input_ids=text_ids, target="graph", source="text")
        syn_text_ids = self.predict(input_ids=graph_ids, target="text", source="graph")
        g2t_outputs = self.teach_model_one_step(
            syn_graph_ids, text_ids, target="text", source="graph"
        )
        t2g_outputs = self.teach_model_one_step(
            syn_text_ids, graph_ids, target="graph", source="text"
        )

        # total loss
        # loss = (
        #     text_outputs.loss + graph_outputs.loss + g2t_outputs.loss + t2g_outputs.loss
        # )
        return {
            "text": text_outputs,
            "graph": graph_outputs,
            "t2g": t2g_outputs,
            "g2t": g2t_outputs,
        }

    def compute_loss_unsup_vae_dual(
        self, text_ids: torch.Tensor, graph_ids: torch.Tensor
    ):
        # -- auto loss (regular VAE)
        text_outputs = self.teach_model_one_step(text_ids, text_ids, target="text")
        graph_outputs = self.teach_model_one_step(graph_ids, graph_ids, target="graph")

        # -- cycle loss (dual)
        syn_graph_ids = self.predict(input_ids=text_ids, target="graph")
        syn_text_ids = self.predict(input_ids=graph_ids, target="text")
        # we need retain_graph=True since in the next step we'll backward through vae_z again
        g2t_outputs = self.teach_model_one_step(
            syn_graph_ids, text_ids, target="text", retain_graph=True
        )
        g2t_outputs_bis = self.teach_model_one_step_with_vae_z(
            syn_graph_ids, target="graph", vae_z=g2t_outputs.vae_latent
        )
        t2g_outputs = self.teach_model_one_step(
            syn_text_ids, graph_ids, target="graph", retain_graph=True
        )
        t2g_outputs_bis = self.teach_model_one_step_with_vae_z(
            syn_text_ids, target="text", vae_z=t2g_outputs.vae_latent
        )

        # total loss
        # loss = (
        #     text_outputs.loss
        #     + graph_outputs.loss
        #     # with the same z~q(z|y_hat)
        #     + g2t_outputs.loss  # log p(x|z) - KL(q(z|y_hat) || p(z))
        #     + g2t_outputs_bis.recon_loss  # log p(y_hat|z)
        #     # idem, with the same z~q(z|x_hat)
        #     + t2g_outputs.loss
        #     + t2g_outputs_bis.recon_loss
        # )
        return {
            "text": text_outputs,
            "graph": graph_outputs,
            "t2g": t2g_outputs,
            "t2g_bis": t2g_outputs_bis,
            "g2t": g2t_outputs,
            "g2t_bis": g2t_outputs_bis,
        }

    def compute_loss_unsup_style_vae(
        self, text_ids: torch.Tensor, graph_ids: torch.Tensor
    ):
        # -- auto loss (style VAE)
        noisy_text_ids = self.get_noisy_inputs(text_ids, is_graph=False)
        noisy_graph_ids = self.get_noisy_inputs(graph_ids, is_graph=True)
        text_outputs = self.teach_model_one_step(
            noisy_text_ids, text_ids, source="text", target="text"
        )
        graph_outputs = self.teach_model_one_step(
            noisy_graph_ids, graph_ids, source="graph", target="graph"
        )

        # -- cycle loss (style VAE)
        syn_graph_ids = self.predict(input_ids=text_ids, target="graph")
        syn_text_ids = self.predict(input_ids=graph_ids, target="text")
        g2t_outputs = self.teach_model_one_step(
            syn_graph_ids, text_ids, source="graph", target="text"
        )
        t2g_outputs = self.teach_model_one_step(
            syn_text_ids, graph_ids, source="text", target="graph"
        )

        return {
            "text": text_outputs,
            "graph": graph_outputs,
            "t2g": t2g_outputs,
            "g2t": g2t_outputs,
        }

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
                if global_step >= self.max_training_steps * (epoch + 1):
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
                outputs = {}
                syn_text_ids = None  # synthetic text and graphs
                syn_graph_ids = None
                noisy_text_ids = None
                noisy_graph_ids = None

                # select beta
                if self.use_vae and self.use_cyclical_beta_schedule:
                    model = self.accelerator.unwrap_model(self.ddp_model)
                    model.beta = self.betas[global_step]

                if self.mode == Mode.t2g:
                    outputs["t2g"] = self.teach_model_one_step(
                        input_ids=text_ids,
                        label_ids=graph_ids,
                        target="graph",
                    )
                elif self.mode == Mode.g2t:
                    outputs["g2t"] = self.teach_model_one_step(
                        input_ids=graph_ids,
                        label_ids=text_ids,
                        target="text",
                    )
                elif self.mode == Mode.both_sup:
                    outputs["g2t"] = self.teach_model_one_step(
                        input_ids=graph_ids,
                        label_ids=text_ids,
                        target="text",
                    )
                    outputs["t2g"] = self.teach_model_one_step(
                        input_ids=text_ids,
                        label_ids=graph_ids,
                        target="graph",
                    )
                elif self.mode == Mode.both_unsup:
                    # todo: make sure the predictions are correctly formatted, especially the attention mask
                    #   -> does it start with an unnecessary padding token?
                    if self.vae_model == VAEModel.non_vae:
                        outputs = self.compute_loss_unsup_non_vae(
                            text_ids=text_ids, graph_ids=graph_ids
                        )
                    elif self.vae_model == VAEModel.full_vae:
                        if self.vae_cycle_loss == CycleVAELoss.single:
                            outputs = self.compute_loss_unsup_vae_single(
                                text_ids=text_ids, graph_ids=graph_ids
                            )
                        elif self.vae_cycle_loss == CycleVAELoss.dual:
                            outputs = self.compute_loss_unsup_vae_dual(
                                text_ids=text_ids, graph_ids=graph_ids
                            )
                    elif (
                        self.vae_model == VAEModel.style_vae
                        or self.vae_model == VAEModel.added_style_vae
                    ):
                        outputs = self.compute_loss_unsup_style_vae(
                            text_ids=text_ids, graph_ids=graph_ids
                        )
                    else:
                        raise ValueError

                # loss.backward has already been called (in teach_model_one_step)
                self.accelerator.clip_grad_norm_(
                    self.ddp_model.parameters(), self.max_grad_norm
                )
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
                global_step += 1

                # log training info (metrics and text sequences)
                self.log_metrics(global_step, epoch, model_outputs=outputs)
                self.log_training_samples(
                    global_step,
                    epoch,
                    text_ids=text_ids,
                    graph_ids=graph_ids,
                    syn_text_ids=syn_text_ids,
                    syn_graph_ids=syn_graph_ids,
                    noisy_text_ids=noisy_text_ids,
                    noisy_graph_ids=noisy_graph_ids,
                    t2g_logits=outputs["t2g"].logits if "t2g" in outputs else None,
                    g2t_logits=outputs["g2t"].logits if "g2t" in outputs else None,
                )

            # free GPU memory before eval
            outputs = {}
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

    def log_metrics(self, global_step: int, epoch: int, model_outputs: dict):
        metrics = {
            "train/learning_rate": self.lr_scheduler.get_last_lr()[0],
            "train/epoch": epoch,
        }
        for mode, outputs in model_outputs.items():
            # for each mode (t2g, g2t, text, ...), log our regular and vae metrics
            outputs = model_outputs[mode]
            metrics[f"train/loss_{mode}"] = outputs.loss.item()
            if "recon_loss" in outputs:
                metrics[f"train/recon_loss_{mode}"] = outputs.recon_loss.item()
            if "reg_loss" in outputs and isinstance(outputs.reg_loss, torch.Tensor):
                metrics[f"train/reg_loss_{mode}"] = outputs.reg_loss.item()

        # log vae_beta coeff if we have a VAE model
        model = self.accelerator.unwrap_model(self.ddp_model)
        if hasattr(model, "beta"):
            metrics["train/beta_t"] = model.beta

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
