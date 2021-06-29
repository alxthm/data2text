import logging
from collections import Counter
from pathlib import Path
from typing import Union, Set, Tuple, Dict

import mlflow
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import (
    default_data_collator,
    PreTrainedTokenizer,
)

from src.data.datasets import Seq2seqDataset, WebNLG2020
from src.data.formatting import Example, Mode
from src.eval.webnlg_g2t.eval import run as run_webnlg_g2t_eval
from src.eval.utils import get_precision_recall_f1
from src.utils import MyLogger


class EvaluatorWebNLG:
    def __init__(
        self,
        mode: Mode,
        datasets: Dict[str, Seq2seqDataset],
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
        self.datasets = datasets
        self.tokenizer = tokenizer
        self.model = model
        self.device = model.device

        self.batch_size = batch_size
        self.num_beams_t2g = num_beams_t2g
        self.num_beams_g2t = num_beams_g2t
        self.max_output_length = datasets["dev"].max_seq_length

        self.logger = MyLogger(
            tensorboard_writer=tensorboard_writer, log_every_n_steps=1
        )
        self.log_path = log_path
        self.limit_samples = limit_samples  # do not use all entire validation dataset

    def evaluate_dev(self, epoch: int):
        self.evaluate_and_log(epoch, split="dev")

    def evaluate_test(self):
        self.evaluate_and_log(0, split="test_all")
        self.evaluate_and_log(0, split="test_seen")
        self.evaluate_and_log(0, split="test_unseen_ent")
        self.evaluate_and_log(0, split="test_unseen_cat")

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
        res, logs = self.run_evaluation(split, epoch)

        # print and save eval metrics
        logging.info(f"[ep{epoch}] eval results: {res}")
        self.logger.log_metrics(metrics=res, step=epoch)

        # save predictions logs to mlflow
        mode = self.mode.value  # t2g, g2t, ...
        file_path = self.log_path / f"{mode}_{split}_{epoch}.txt"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(logs)
        mlflow.log_artifact(str(file_path), f"{mode}_out/{epoch}")

    def run_evaluation(self, split: str, epoch: int):
        """
        Evaluate model on this dataset. Do inference, compute and return metrics
        """
        dataset = self.datasets[split]

        if self.mode == Mode.t2g:
            return self.run_evaluation_t2g(dataset, split)
        elif self.mode == Mode.g2t:
            return self.run_evaluation_g2t(dataset, split, epoch=epoch)
        else:
            raise ValueError

    def run_evaluation_g2t(self, dataset, split: str, epoch: int):
        dataloader = DataLoader(
            Subset(dataset, dataset.unique_graph_ids),
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=default_data_collator,
        )
        text_predictions = []
        logs = ""
        for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            if self.limit_samples and (i + 1) * self.batch_size > self.limit_samples:
                # to speed up validation, do not consider all samples
                break
            batch_preds, batch_logs = self.make_pred_g2t_batch(batch, dataset)
            text_predictions += batch_preds
            logs += batch_logs

        # save predictions to a txt file
        hyps_path = self.log_path / f"text_pred_{self.mode.value}_{split}_{epoch}.txt"
        with open(hyps_path, "w", encoding="utf-8") as f:
            f.write("\n".join(text_predictions))

        data_dir = self.log_path / "../../data"
        refs_path = data_dir / f"processed/{dataset.dataset_name}/ref/{split}_<id>.txt"
        metrics = run_webnlg_g2t_eval(
            refs_path=str(refs_path.resolve()),
            hyps_path=hyps_path,
            num_refs=4,
            lng="en",
            metrics="bleu,meteor,bert",
        )
        return metrics, logs

    def make_pred_g2t_batch(self, batch, dataset: WebNLG2020):
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

        # log predictions and reference texts
        logs = ""
        assert len(text_predictions) == len(references)
        for example_id, text, ref in zip(
            batch["example_id"], text_predictions, references
        ):
            example = dataset.get_example(example_id)
            logs += (
                f"[{example_id}] input graph / output / label / (raw)\n"
                f"{example.graph}\n"
                f"{text}\n"
                f"{ref}\n"
                f"({example.text})\n"
            )

        return text_predictions, logs

    def run_evaluation_t2g(self, dataset, split: str):
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,  # if using shuffle and dataset.get_example(id), make sure that id is correct
            collate_fn=default_data_collator,
        )

        t2g_results = Counter()
        logs = ""
        for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            if self.limit_samples and (i + 1) * self.batch_size > self.limit_samples:
                # to speed up validation, do not consider all samples
                break
            batch_results, batch_logs = self.evaluate_t2g_batch(batch, dataset)
            t2g_results += batch_results
            logs += batch_logs

        metrics = self.compute_t2g_metrics(t2g_results)
        metrics = {f"{split}/{k}": v for k, v in metrics.items()}
        return metrics, logs

    def evaluate_t2g_batch(self, batch, dataset: WebNLG2020):
        batch_results = Counter()
        logs = ""

        # get raw batch predictions
        graph_predictions = self.model.generate(
            batch["text_ids"].to(self.device),
            max_length=self.max_output_length,
            num_beams=self.num_beams_t2g,
        )

        for example_id, text_ids, graph_ids, graph_prediction in zip(
            batch["example_id"],
            batch["text_ids"],
            batch["graph_ids"],
            graph_predictions,
        ):
            # decode the token ids (of prediction, label and input)
            example = dataset.get_example(example_id)
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

            # log example
            logs += (
                f"[{example_id}] input / output / label (+pred/gt rel)\n"
                f"{text_in_sentence}\n"
                f"{graph_out_sentence}\n"
                f"{graph_label_sentence}\n"
            )
            if error_log:
                # predicted graph did not match exactly target graph
                new_results["graph_errors"] += 1
                # log predicted vs ground truth sets of relations
                logs += error_log

            # update statistics: number of correct/pred/gt relations, etc
            batch_results += new_results

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

    # -- g2t metrics using huggingface library (sacrebleu,etc)
    #   (could be useful if using other datasets than webnlg, with only one lexicalization)
    #
    # # use sacrebleu (a standardized version of BLEU, using a standard tokenizer)
    # # https://github.com/huggingface/datasets/issues/137
    # g2t_metrics = {"bleu": load_metric("sacrebleu"), "rouge": load_metric("rouge")}
    #
    # def evaluate_g2t_batch(
    #     self, batch, batch_id: int, g2t_metrics: Dict[str, Metric], dataset: Seq2seqDataset
    # ):
    #     # get raw batch predictions
    #     text_predictions_ids = self.model.generate(
    #         batch["graph_ids"].to(self.device),
    #         max_length=self.max_output_length,
    #         num_beams=self.num_beams_g2t,
    #     )
    #     # transform predictions and references to plain text (detokenized)
    #     text_predictions = self.tokenizer.batch_decode(
    #         text_predictions_ids, skip_special_tokens=True
    #     )
    #     references = self.tokenizer.batch_decode(
    #         batch["text_ids"], skip_special_tokens=True
    #     )
    #     # for each input text, a list of reference texts is expected for sacrebleu
    #     g2t_metrics["bleu"].add_batch(
    #         predictions=text_predictions, references=[[r] for r in references]
    #     )
    #     g2t_metrics["rouge"].add_batch(
    #         predictions=text_predictions, references=references
    #     )
    #
    #     # log predictions and reference texts
    #     logs = ""
    #     assert len(text_predictions) == len(references)
    #     for j in range(len(text_predictions)):
    #         current_id = self.batch_size * batch_id + j
    #         example = dataset.get_example(current_id)
    #         logs += (
    #             f"[{current_id}] input graph / output / label / (raw)\n"
    #             f"{example.graph}\n"
    #             f"{text_predictions[j]}\n"
    #             f"{references[j]}\n"
    #             f"({example.text})\n"
    #         )
    #     return logs
    #
    # def compute_g2t_metrics(self, g2t_metrics: Dict[str, Metric]):
    #     bleu = g2t_metrics["bleu"]
    #     n = len(bleu)  # number of stored examples in the metric
    #     bleu_results = bleu.compute()
    #     rouge_results = g2t_metrics["rouge"].compute()
    #     # to match cyclegt evaluation, compute mean F1 score of rougeL metric
    #     # (mid is the mean, see https://github.com/google-research/google-research/blob/master/rouge/scoring.py#L141)
    #     rouge_score = rouge_results["rougeL"].mid.fmeasure
    #     return {
    #         "bleu": bleu_results["score"],
    #         "rouge_l": rouge_score,
    #         # sys/ref_len is the length (in space separated tokens, i.e. words) of predicted/label sentences
    #         # https://github.com/mjpost/sacrebleu/blob/5dfcaa3cee00039bcad7a12147b6d6976cb46f42/sacrebleu/metrics/bleu.py#L248
    #         "avg_predicted_len": bleu_results["sys_len"] / n,
    #         "avg_correct_len": bleu_results["ref_len"] / n,
    #     }
    #