import logging
from pathlib import Path
from typing import Union

import mlflow
from sklearn.metrics import f1_score, accuracy_score
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import (
    default_data_collator,
    TrainerState,
    TrainerCallback,
    TrainerControl,
    TrainingArguments,
)

from src.data.datasets import WebNLG
from src.utils import MyLogger


class EvalCallback(TrainerCallback):
    """
    Run evaluation on val dataset on epoch end, and eval on test when training is done

    (not necessary when using custom training loop, instead of the default
    huggingface Trainer class)
    """

    def __init__(self, evaluator: "Evaluator"):
        self.evaluator = evaluator

    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        self.evaluator.evaluate_and_log(state.epoch, split="val")

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        self.evaluator.evaluate_and_log(0, split="test")


class Evaluator:
    def __init__(
        self,
        val_dataset: WebNLG,
        test_dataset: WebNLG,
        tokenizer,
        model,
        batch_size: int,
        num_beams: int,
        log_path: Path,
        tensorboard_writer: SummaryWriter = None,
        limit_samples: Union[int, bool] = False,
    ):
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.tokenizer = tokenizer
        self.model = model
        self.device = model.device

        self.batch_size = batch_size
        self.num_beams = num_beams
        self.max_output_length = val_dataset.max_output_length

        self.logger = MyLogger(tensorboard_writer)
        self.log_path = log_path
        self.limit_samples = limit_samples  # do not use all entire validation dataset

    def evaluate_and_log(self, epoch, split: str):
        """
        Evaluate model on val or test dataset.
        Log metrics to tensorboard & mlflow, log artifacts (input/predictions/labels)
        to mlflow as txt files
        """
        epoch = int(epoch)

        # get model ready for inference
        self.model.eval()  # .no_grad() already called by model.generate(), but not .eval()
        self.model.to(self.device)

        # run the evaluation loop and get resulting metrics
        logging.info(f"[ep{epoch}] evaluating on {split}...")
        res, logs = self.run_evaluation(split)

        # print and save eval metrics
        logging.info(f"[ep{epoch}] eval results: {res}")
        self.logger.log_metrics(metrics=res, step=epoch)

        # save predictions logs to mlflow
        with open(self.log_path / f"t2g_{split}_{epoch}.txt", "w", encoding="utf-8") as f:
            f.write(logs)
        mlflow.log_artifact(str(self.log_path / f"t2g_{split}_{epoch}.txt"), f"t2g_out/{epoch}")

    def run_evaluation(self, split: str):
        """
        Evaluate model on this dataset. Do inference, compute and return metrics
        """
        if split == "val":
            dataset = self.val_dataset
        elif split == "test":
            dataset = self.test_dataset
        else:
            raise ValueError

        (
            entities_pred,
            entities_true,
            relations_pred,
            relations_true,
            format_errors,
            logs,
        ) = self.get_predictions(dataset)

        # compute and return scores
        entity_f1_micro = f1_score(entities_true, entities_pred, average="micro")
        entity_f1_macro = f1_score(entities_true, entities_pred, average="macro")
        entity_acc = accuracy_score(entities_true, entities_pred)
        relation_f1_micro = f1_score(relations_true, relations_pred, average="micro")
        relation_f1_macro = f1_score(relations_true, relations_pred, average="macro")
        relation_acc = accuracy_score(relations_true, relations_pred)

        res = {
            f"{split}/entity_acc": entity_acc,
            f"{split}/entity_f1_micro": entity_f1_micro,
            f"{split}/entity_f1_macro": entity_f1_macro,
            f"{split}/relation_acc": relation_acc,
            f"{split}/relation_f1_micro": relation_f1_micro,
            f"{split}/relation_f1_macro": relation_f1_macro,
            # % errors when parsing model output
            f"{split}/format_error": format_errors / len(entities_true),
        }

        return res, logs

    def get_predictions(self, dataset: WebNLG):
        """
        Do inference on dataset, and return predicted & ground truth relations/entities
        """
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=default_data_collator,
        )

        format_errors = 0
        entities_pred = []
        entities_true = []
        relations_pred = []
        relations_true = []
        logs = ""

        for i, inputs in tqdm(enumerate(dataloader), total=len(dataloader)):
            if self.limit_samples and len(entities_true) > self.limit_samples:
                # to speed up validation, do not consider all samples
                break

            # get raw batch predictions
            predictions = self.model.generate(
                inputs["input_ids"].to(self.device),
                max_length=self.max_output_length,
                num_beams=self.num_beams,
            )

            for j, (input_ids, label_ids, prediction) in enumerate(
                zip(inputs["input_ids"], inputs["labels"], predictions)
            ):
                # decode the token ids (of prediction, label and input)
                current_id = i * self.batch_size + j
                example = dataset.get_example(current_id)
                output_sentence = self.tokenizer.decode(
                    prediction,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                input_sentence = self.tokenizer.decode(
                    input_ids,
                    skip_special_tokens=True,
                )
                label_sentence = self.tokenizer.decode(
                    label_ids,
                    skip_special_tokens=True,
                )

                # parse output sequence, obtain sets of predicted entities and relations
                parsed_output = dataset.output_format.run_inference(output_sentence)
                predicted_entities, predicted_relations, wrong_format = parsed_output
                entities_pred.append(str(predicted_entities))
                relations_pred.append(str(predicted_relations))

                # load ground truth entities and relations (as str and tuple of str)
                gt_entities = set()
                gt_relations = set()
                for relation in example.graph:
                    gt_entities.update({relation.head.text, relation.tail.text})
                    gt_relations.add(
                        (relation.head.text, relation.type.natural, relation.tail.text)
                    )
                # use a set so the order of entities/relations does not matter
                # and then convert to str (with a determined order) for evaluation
                entities_true.append(str(gt_entities))
                relations_true.append(str(gt_relations))

                if wrong_format:
                    format_errors += 1

                # text to log
                logs += (
                    f"[{current_id}] input / output / label (+pred/gt rel)\n"
                    f"{input_sentence}\n"
                    f"{output_sentence}\n"
                    f"{label_sentence}\n"
                )
                if predicted_relations != gt_relations:
                    logs += "error\n" f"{predicted_relations}\n" f"{gt_relations}\n"

        return (
            entities_pred,
            entities_true,
            relations_pred,
            relations_true,
            format_errors,
            logs,
        )
