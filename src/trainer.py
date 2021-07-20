import logging
import random

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
from src.data.noise_functions import noise_functions_list
from src.eval.evaluator import EvaluatorWebNLG
from src.utils import MyLogger, Mode


class Seq2seqTrainer:
    # to be set after trainer init (we need to create Trainer with accelerator first)
    evaluator: EvaluatorWebNLG

    def __init__(
        self,
        model,
        mode: Mode,
        tokenizer: PreTrainedTokenizer,
        train_dataset: Seq2seqDataset,
        accelerator: Accelerator,
        learning_rate: float,
        batch_size: int,
        num_epochs: int,
        tensorboard_writer: SummaryWriter,
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
        (
            self.ddp_model,
            self.optimizer,
            self.train_dataloader,
        ) = accelerator.prepare(model, optimizer, train_dataloader)

        # training parameters
        if max_training_steps > 0:
            self.num_training_steps = max_training_steps
            self.num_epochs = 1
        else:
            self.num_training_steps = num_epochs * len(self.train_dataloader)
            self.num_epochs = num_epochs
        self.lr_scheduler = get_scheduler(
            "linear",
            optimizer=self.optimizer,
            num_warmup_steps=0,
            num_training_steps=self.num_training_steps,
        )
        self.batch_size = batch_size
        self.max_seq_length = train_dataset.max_seq_length
        self.max_grad_norm = max_grad_norm

        # logging
        self.logger = MyLogger(
            tensorboard_writer=tensorboard_writer,
            log_every_n_steps=log_every_n_steps,
            accelerator=accelerator,
            use_loggers=tensorboard_writer is not None,
        )

    def set_evaluator(self, evaluator: EvaluatorWebNLG):
        self.evaluator = evaluator

    def predict(self, input_ids: torch.Tensor, target: str):
        input_ids = add_prefix(
            input_ids=input_ids,
            target=target,
            tokenizer=self.tokenizer,
            max_seq_len=self.max_seq_length,
        )
        self.ddp_model.eval()
        with torch.no_grad():
            model = self.accelerator.unwrap_model(self.ddp_model)
            # todo: try sampling instead of greedy ? better results?
            prediction_ids = model.generate(
                input_ids,
                max_length=self.max_seq_length,
                num_beams=1,
            )
        # multi-GPU: no need to gather predictions across processes yet, since the
        # predictions are to be used in training (gathering is down after the loss is computed)
        return prediction_ids

    def teach_model_one_step(
        self, input_ids: torch.Tensor, label_ids: torch.Tensor, target: str
    ):
        """

        Args:
            input_ids: input sequence (text/graph tokenized batch, already on device, with prefix)
            label_ids: label (ground truth graph/text as a tokenized sequence)
            target: 'text' or 'graph', depending on the format of label sequences. Will
                determine the prefix to add to input_ids

        Returns:
            loss

        """
        input_ids = add_prefix(
            input_ids=input_ids,
            target=target,
            tokenizer=self.tokenizer,
            max_seq_len=self.max_seq_length,
        )
        att_mask_input = self.get_att_mask(input_ids)
        self.ddp_model.train()
        outputs = self.ddp_model(
            input_ids=input_ids,
            attention_mask=att_mask_input,
            labels=label_ids,
        )
        loss = outputs.loss
        self.accelerator.backward(loss)
        return loss

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
                if global_step > self.num_training_steps:
                    break

                # get batch data
                text_ids = batch["text_ids"]
                graph_ids = batch["graph_ids"]
                att_mask_text = batch["att_mask_text"]
                att_mask_graph = batch["att_mask_graph"]
                assert (att_mask_graph == self.get_att_mask(graph_ids)).all()
                assert (att_mask_text == self.get_att_mask(text_ids)).all()

                # training step
                loss_g2t = torch.tensor(0)
                loss_t2g = torch.tensor(0)
                loss_text = torch.tensor(0)
                loss_graph = torch.tensor(0)
                if self.mode == Mode.t2g:
                    loss_t2g = self.teach_model_one_step(
                        input_ids=text_ids, label_ids=graph_ids, target="graph"
                    )
                elif self.mode == Mode.g2t:
                    loss_g2t = self.teach_model_one_step(
                        input_ids=graph_ids, label_ids=text_ids, target="text"
                    )
                elif self.mode == Mode.both_sup:
                    loss_g2t = self.teach_model_one_step(
                        input_ids=graph_ids, label_ids=text_ids, target="text"
                    )
                    loss_t2g = self.teach_model_one_step(
                        input_ids=text_ids, label_ids=graph_ids, target="graph"
                    )
                elif self.mode == Mode.both_unsup:
                    # todo: make sure the predictions are correctly formatted, especially the attention mask
                    #   -> does it start with an unnecessary padding token?
                    #   -> should we hide the prefix ("Generate graph/text:") to the encoder in the input?

                    # todo: samples will be correlated with denoising autoencoding+backtranslation
                    #   -> do one epoch of each like GT-BT?
                    #   -> or see what Lample does?
                    # text denoising auto-encoder step
                    noisy_text_ids = self.get_noisy_inputs(text_ids, is_graph=False)
                    loss_text = self.teach_model_one_step(
                        input_ids=noisy_text_ids, label_ids=text_ids, target="text"
                    )
                    # graph denoising auto-encoder step
                    noisy_graph_ids = self.get_noisy_inputs(graph_ids, is_graph=False)
                    loss_graph = self.teach_model_one_step(
                        input_ids=noisy_graph_ids, label_ids=graph_ids, target="graph"
                    )
                    # g2t unsupervised step
                    graph_pred_ids = self.predict(input_ids=text_ids, target="graph")
                    loss_g2t = self.teach_model_one_step(
                        input_ids=graph_pred_ids, label_ids=text_ids, target="text"
                    )
                    # t2g unsupervised step
                    text_pred_ids = self.predict(input_ids=graph_ids, target="text")
                    loss_t2g = self.teach_model_one_step(
                        input_ids=text_pred_ids, label_ids=graph_ids, target="graph"
                    )
                else:
                    raise ValueError

                self.accelerator.clip_grad_norm_(
                    self.ddp_model.parameters(), self.max_grad_norm
                )
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
                global_step += 1

                # log training info
                self.logger.log_metrics(
                    {
                        "train/loss_text": loss_text.item(),
                        "train/loss_graph": loss_graph.item(),
                        "train/loss_t2g": loss_t2g.item(),
                        "train/loss_g2t": loss_g2t.item(),
                        "train/learning_rate": self.lr_scheduler.get_last_lr()[0],
                        "train/epoch": epoch,
                    },
                    step=global_step,
                )

            # evaluate after each epoch (and save model checkpoint if necessary)
            self.evaluator.on_epoch_end(epoch)

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
            noise_fun = random.choice(noise_functions_list)
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
