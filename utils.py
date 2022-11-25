from datasets import load_metric
import numpy as np
import json
import torch
from pathlib import Path
from tqdm import tqdm
import time
from transformers import EvalPrediction


def records_to_columns(records):
    keys = records[0].keys()
    columns = {}
    for key in keys:
        columns[key] = [record[key] for record in records]
    return columns


def tokenize_and_align_labels(examples, tokenizer, label_all_tokens=True):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        max_length=512,
    )

    labels = []
    for i, label in enumerate(examples[f"ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def align_tokens_to_annos(encoding, annos, label_all_tokens):
    annos.sort(key=lambda x: x["start"])

    anno_ix = 0
    results = []
    done = len(annos) == 0
    prev_word_id = -1
    for word_id, offset in zip(encoding.word_ids, encoding.offsets):

        if word_id is None:
            results.append(-100)
            continue

        if done == True:
            results.append(0)
        else:
            anno = annos[anno_ix]
            start, end = offset

            if end < anno["start"]:
                # the offset is before the next annotation
                results.append(0)
            elif start <= anno["start"] and end <= anno["end"]:
                results.append(1)
            elif start >= anno["start"] and end <= anno["end"]:
                is_same_word = prev_word_id == word_id
                results.append(
                    -100 if (not label_all_tokens and is_same_word) else 2
                )
            elif start >= anno["start"] and end > anno["end"]:
                anno_ix += 1
                results.append(0)
            else:
                raise Exception(
                    f"Funny Overlap {offset},{anno}",
                )

            if anno_ix >= len(annos):
                done = True
            prev_word_id = word_id

    return results


def corpus_to_encodings(corp, tokenizer, label_all_tokens=True):
    encodings = tokenizer(
        [s["text"] for s in corp], truncation=True, max_length=512
    )
    encodings["labels"] = []
    # encodings["text"] = []

    for i, doc in enumerate(corp):
        labels = align_tokens_to_annos(
            encodings[i], doc["spans"], label_all_tokens
        )
        encodings["labels"].append(labels)
        # encodings["text"].append(doc["text"])

    return encodings


metrics = load_metric("seqeval")

label_list = {0: "O", 1: "B", 2: "I"}


def compute_mtner_metrics(eval_pred: EvalPrediction):
    predictions = eval_pred.predictions
    labels = eval_pred.label_ids
    entity_type = eval_pred.inputs["entity_type"]
    entity_type_ids = np.sort(np.unique(entity_type))

    metrics = {}
    for entity_idx in entity_type_ids:
        # in the entire batch, these indices contain the samples
        # corresponding to this entity type
        indices = np.argwhere(entity_type == entity_idx)

        # predictions.shape = (batch_size, num_entity_types, seq_len,
        # num_features)
        preds = predictions[indices, entity_idx, :, :]
        # (batch_size, seq_len, num_features)
        preds = np.squeeze(preds, axis=1)

        lab = labels[indices, entity_idx, :, :]
        lab = np.squeeze(lab, axis=1)

        metrics[entity_idx] = compute_metrics((preds, lab))

    return metrics


def compute_metrics(p):
    label_list = {0: "O", 1: "B", 2: "I"}

    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metrics.compute(
        predictions=true_predictions, references=true_labels
    )
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
        "corpus_size": len(true_labels),
    }


class NerDataset(torch.utils.data.Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples["labels"])

    def __getitem__(self, idx):
        item = {
            key: val[idx]
            for key, val in self.examples.items()
            if key != "token_type_ids"
        }
        return item


def json_to_torch_dataset(path, tokenizer):
    with open(path, "r") as f:
        examples = json.load(f)
    return NerDataset(corpus_to_encodings(examples, tokenizer))


# UTILITIES FOR COMPRESSION


def evaluate(model, dataset: NerDataset):
    """
    model: token classification / NER model
    dataset: torch dataset with each record a dict-like object with 'input_ids', 'labels', 'attention_mask'
    """
    preds = []
    labels = []
    with torch.no_grad():
        for i in tqdm(range(len(dataset))):
            pred = model(
                torch.tensor(dataset[i]["input_ids"]).unsqueeze(dim=0)
            )[0]
            pred = pred.numpy().argmax(axis=2).squeeze()
            label = dataset[i]["labels"]
            assert len(pred) == len(label)

            preds.append(pred)
            labels.append(dataset[i]["labels"])

    return compute_metrics_alt(preds, labels)


def compute_size(model):
    """This overrides the PerformanceBenchmark.compute_size() method"""
    state_dict = model.state_dict()
    tmp_path = Path("model.pt")
    torch.save(state_dict, tmp_path)
    # Calculate size in megabytes
    size_mb = Path(tmp_path).stat().st_size / (1024 * 1024)
    # Delete temporary file
    tmp_path.unlink()
    print(f"Model size (MB) - {size_mb:.2f}")
    return {"size_mb": size_mb}


def compute_metrics_alt(predictions, labels):
    label_list = {0: "O", 1: "B", 2: "I"}

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metrics.compute(
        predictions=true_predictions, references=true_labels
    )
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
        "corpus_size": len(true_labels),
    }
