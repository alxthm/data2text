import logging
from collections import Counter
from pathlib import Path
from typing import Union, Set, Tuple, Dict

import mlflow
from datasets import load_metric, Metric
from sklearn.metrics import f1_score, accuracy_score
from tokenizers import Tokenizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import (
    default_data_collator,
    TrainerState,
    TrainerCallback,
    TrainerControl,
    TrainingArguments,
    PreTrainedTokenizer,
)

from src.data.datasets import WebNLG
from src.data.formatting import Example, Mode
from src.utils import MyLogger, get_precision_recall_f1


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
        mode: Mode,
        val_dataset: WebNLG,
        test_dataset: WebNLG,
        tokenizer: PreTrainedTokenizer,
        model,
        batch_size: int,
        num_beams_t2g: int,
        num_beams_g2t: int,
        log_path: Path,
        tensorboard_writer: SummaryWriter = None,
        limit_samples: Union[int, bool] = False,
    ):
        self.mode = mode
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.tokenizer = tokenizer
        self.model = model
        self.device = model.device

        self.batch_size = batch_size
        self.num_beams_t2g = num_beams_t2g
        self.num_beams_g2t = num_beams_g2t
        self.max_output_length = val_dataset.max_seq_length

        self.logger = MyLogger(
            tensorboard_writer=tensorboard_writer, log_every_n_steps=1
        )
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
        mode = self.mode.value  # t2g, g2t, ...
        file_path = self.log_path / f"{mode}_{split}_{epoch}.txt"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(logs)
        mlflow.log_artifact(str(file_path), f"{mode}_out/{epoch}")

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

        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,  # if using shuffle and dataset.get_example(id), make sure that id is correct
            collate_fn=default_data_collator,
        )

        t2g_results = Counter()
        # use sacrebleu (a standardized version of BLEU, using a standard tokenizer)
        # https://github.com/huggingface/datasets/issues/137
        g2t_metrics = {"bleu": load_metric("sacrebleu"), "rouge": load_metric("rouge")}
        logs = ""

        for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            if self.limit_samples and (i + 1) * self.batch_size > self.limit_samples:
                # to speed up validation, do not consider all samples
                break

            if self.mode == Mode.t2g:
                batch_results, batch_logs = self.evaluate_t2g_batch(batch, i, dataset)
                t2g_results += batch_results
            elif self.mode == Mode.g2t:
                # update metrics with batch predictions and reference texts
                batch_logs = self.evaluate_g2t_batch(batch, i, g2t_metrics, dataset)
            else:
                raise ValueError
            logs += batch_logs

        if self.mode == Mode.t2g:
            metrics = self.compute_t2g_metrics(t2g_results)
        elif self.mode == Mode.g2t:
            metrics = self.compute_g2t_metrics(g2t_metrics)
        else:
            raise ValueError

        metrics = {f"{split}/{k}": v for k, v in metrics.items()}
        return metrics, logs

    def compute_t2g_metrics(self, t2g_results):
        entity_precision, entity_recall, entity_f1 = get_precision_recall_f1(
            num_correct=t2g_results["correct_entities"],
            num_predicted=t2g_results["predicted_entities"],
            num_gt=t2g_results["gt_entities"],
        )
        relation_precision, relation_recall, relation_f1 = get_precision_recall_f1(
            num_correct=t2g_results["correct_relations"],
            num_predicted=t2g_results["predicted_relations"],
            num_gt=t2g_results["gt_relations"],
        )
        n = t2g_results["num_sentences"]
        metrics = {
            f"entity_f1": entity_f1,
            f"entity_precision": entity_precision,
            f"entity_recall": entity_recall,
            f"relation_f1": relation_f1,
            f"relation_precision": relation_precision,
            f"relation_recall": relation_recall,
            # % errors when parsing model output
            f"format_error": t2g_results["wrong_format"] / n,
            f"graph_acc": (n - t2g_results["graph_errors"]) / n,
        }
        return metrics

    def evaluate_t2g_batch(self, batch, batch_id, dataset: WebNLG):
        batch_results = Counter()
        logs = ""

        # get raw batch predictions
        graph_predictions = self.model.generate(
            batch["text_ids"].to(self.device),
            max_length=self.max_output_length,
            num_beams=self.num_beams_t2g,
        )

        for j, (text_ids, graph_ids, graph_prediction) in enumerate(
            zip(batch["text_ids"], batch["graph_ids"], graph_predictions)
        ):
            # decode the token ids (of prediction, label and input)
            current_id = batch_id * self.batch_size + j
            example = dataset.get_example(current_id)
            graph_out_sentence = self.tokenizer.decode(
                graph_prediction,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            text_in_sentence = self.tokenizer.decode(text_ids, skip_special_tokens=True)
            graph_label_sentence = self.tokenizer.decode(
                graph_ids, skip_special_tokens=True
            )

            # parse sentence with predicted graph, obtain sets of predicted entities and relations
            parsed_graph_pred = dataset.graph_format.extract_raw_graph(
                graph_out_sentence
            )
            predicted_entities, predicted_relations, wrong_format = parsed_graph_pred

            # compare predictions to ground truth
            new_results, error_log = self.evaluate_t2g_example(
                example, predicted_entities, predicted_relations
            )
            if wrong_format:
                new_results["wrong_format"] = 1
            # update statistics: number of correct/pred/gt relations, etc
            batch_results += new_results

            # log example
            logs += (
                f"[{current_id}] input / output / label (+pred/gt rel)\n"
                f"{text_in_sentence}\n"
                f"{graph_out_sentence}\n"
                f"{graph_label_sentence}\n"
            )
            if error_log:
                # predicted graph did not match exactly target graph
                new_results["graph_errors"] += 1
                # log predicted vs ground truth sets of relations
                logs += error_log

        return batch_results, logs

    def evaluate_t2g_example(
        self,
        example: Example,
        predicted_entities: Set[str],
        predicted_relations: Set[Tuple[str]],
    ):
        """
        Evaluate model predictions on a single example of this dataset.

        Return the number of predicted/ground-truth/correct relations (and entities) for
        this example. This can then be used to compute f1 scores.

        Ground truth and predicted triples are matched using python sets.
        """
        # load ground truth entities and relations
        gt_entities = set()
        gt_relations = set()
        for relation in example.graph:
            gt_entities.add(relation.head.text)
            gt_entities.add(relation.tail.text)
            gt_relations.add(relation.to_tuple())

        # filter correct entities and relations
        correct_entities = predicted_entities & gt_entities
        correct_relations = predicted_relations & gt_relations

        assert len(correct_entities) <= len(predicted_entities)
        assert len(correct_entities) <= len(gt_entities)
        assert len(correct_relations) <= len(predicted_relations)
        assert len(correct_relations) <= len(gt_relations)

        res = Counter(
            {
                "num_sentences": 1,
                "gt_entities": len(gt_entities),
                "predicted_entities": len(predicted_entities),
                "correct_entities": len(correct_entities),
                "gt_relations": len(gt_relations),
                "predicted_relations": len(predicted_relations),
                "correct_relations": len(correct_relations),
            }
        )

        error_log = None
        if not len(correct_relations) == len(predicted_relations) == len(gt_relations):
            error_log = "error\n" f"{predicted_relations}\n" f"{gt_relations}\n"

        return res, error_log

    def evaluate_g2t_batch(
        self, batch, batch_id: int, g2t_metrics: Dict[str, Metric], dataset: WebNLG
    ):
        # get raw batch predictions
        text_predictions_ids = self.model.generate(
            batch["graph_ids"].to(self.device),
            max_length=self.max_output_length,
            num_beams=self.num_beams_g2t,
        )
        # transform predictions and references to plain text (detokenized)
        text_predictions = self.tokenizer.batch_decode(
            text_predictions_ids, skip_special_tokens=True
        )
        references = self.tokenizer.batch_decode(
            batch["text_ids"], skip_special_tokens=True
        )
        # for each input text, a list of reference texts is expected for sacrebleu
        g2t_metrics["bleu"].add_batch(
            predictions=text_predictions, references=[[r] for r in references]
        )
        g2t_metrics["rouge"].add_batch(
            predictions=text_predictions, references=references
        )

        # log predictions and reference texts
        logs = ""
        assert len(text_predictions) == len(references)
        for j in range(len(text_predictions)):
            current_id = self.batch_size * batch_id + j
            example = dataset.get_example(current_id)
            logs += (
                f"[{current_id}] input graph / output / label / (raw)\n"
                f"{example.graph}\n"
                f"{text_predictions[j]}\n"
                f"{references[j]}\n"
                f"({example.text})\n"
            )
        return logs

    def compute_g2t_metrics(self, g2t_metrics: Dict[str, Metric]):
        bleu = g2t_metrics["bleu"]
        n = len(bleu)  # number of stored examples in the metric
        bleu_results = bleu.compute()
        rouge_results = g2t_metrics["rouge"].compute()
        # to match cyclegt evaluation, compute mean F1 score of rougeL metric
        # (mid is the mean, see https://github.com/google-research/google-research/blob/master/rouge/scoring.py#L141)
        rouge_score = rouge_results["rougeL"].mid.fmeasure
        return {
            "bleu": bleu_results["score"],
            "rouge_l": rouge_score,
            # sys/ref_len is the length (in space separated tokens, i.e. words) of predicted/label sentences
            # https://github.com/mjpost/sacrebleu/blob/5dfcaa3cee00039bcad7a12147b6d6976cb46f42/sacrebleu/metrics/bleu.py#L248
            "avg_predicted_len": bleu_results["sys_len"] / n,
            "avg_correct_len": bleu_results["ref_len"] / n,
        }
