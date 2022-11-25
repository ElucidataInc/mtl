from typing import List
import torch
from torch import nn
import numpy as np
from utils import compute_metrics
from utils import json_to_torch_dataset
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer,
    BertModel,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

pmbert_path = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
pmbert = BertModel.from_pretrained(pmbert_path)
tokenizer = AutoTokenizer.from_pretrained(pmbert_path)


class MultiNER(nn.Module):
    def __init__(self, num_entity_types: int, freeze_base_bert=False):
        super().__init__()
        self.num_entity_types = num_entity_types
        self.base_bert = pmbert
        self.dropout = nn.Dropout()
        classifiers = []
        for i in range(self.num_entity_types):
            classifiers.append(
                nn.Sequential(
                    nn.Linear(768, 768), nn.ReLU(), nn.Linear(768, 3)
                )
            )
        self.classifiers = nn.ModuleList(classifiers)
        if freeze_base_bert:
            for param in self.base_bert.parameters():
                param.requires_grad = False

        # these lines don't affect training
        # I've added these so this model can be used
        # with hf's TokenClassificationPipeline for inference
        self.config = self.base_bert.config
        self.config.id2label = {0: "O", 1: "B", 2: "I"}

    def forward(
        self, input_ids, attention_mask, labels=None, entity_type=None
    ):
        bert_output = self.base_bert(
            input_ids=input_ids, attention_mask=attention_mask
        )
        x = self.dropout(bert_output.last_hidden_state)
        outputs = [classifier(x) for classifier in self.classifiers]

        # output has shape (batch_size, num_entity_types, max_seq_length,
        # num_features)
        return {"logits": torch.stack(outputs, dim=1)}

    @classmethod
    def from_pretrained(cls, num_entity_types, path):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        model = cls(num_entity_types)
        model.load_state_dict(
            torch.load(path, map_location=torch.device(device))
        )
        model.eval()
        return model


class MultiNerDataset(Dataset):
    def __init__(self, ner_datasets):
        """
        ner_dataset - Sequence of ner datasets,
            each for a given entity type.
            This class can be used if one doc only
            has annotations for 1 entity types
        """
        self.num_entity_types = len(ner_datasets)
        self.examples = []
        self.entity_type = []

        for entity_idx, dataset in enumerate(ner_datasets):
            for example in dataset:
                assert "labels" in example

                example["entity_type"] = entity_idx
                self.examples.append(example)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


def _compute_loss(model: MultiNER, inputs, return_outputs=False):
    assert "labels" in inputs.keys() and "entity_type" in inputs.keys()
    output = model(**inputs)
    logits = output["logits"]
    loss_fn = nn.CrossEntropyLoss()
    # (batch_size, num_features, num_entity_types, seq_length)
    logits = logits.transpose(2, 3).transpose(1, 2)
    labels = inputs["labels"]

    # (batch_size, seq_len) -> (batch_size, num_entity_types, seq_len)
    labels = labels.unsqueeze(1).repeat(1, model.num_entity_types, 1)

    # (batch_size)
    entity_type = inputs["entity_type"]

    for entity_idx in range(model.num_entity_types):
        labels[entity_type != entity_idx, entity_idx, :] = -100

    loss = loss_fn(logits, labels)
    return (loss, output) if return_outputs else loss


class MultiNERTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        return _compute_loss(model, inputs, return_outputs)

    def evaluate(
        self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"
    ):
        if eval_dataset is None:
            eval_dataset = self.eval_dataset

        collator = DataCollatorForTokenClassification(
            tokenizer,
            max_length=512,
            padding="max_length",
            label_pad_token_id=-100,
            return_tensors="pt",
        )

        loader = DataLoader(
            eval_dataset,
            batch_size=self.args.per_device_eval_batch_size,
            shuffle=False,
            collate_fn=collator,
            pin_memory=True,
        )

        logits, labels, entity_type = [], [], []
        for batch in tqdm(loader):
            self.model.eval()
            batch = {
                key: val.to(self.args.device) for key, val in batch.items()
            }
            with torch.no_grad():
                logit = self.model(**batch)["logits"]
            logits.append(logit.cpu().numpy())
            labels.append(batch["labels"].cpu().numpy())
            entity_type.append(batch["entity_type"].cpu().numpy())

        logits, labels, entity_type = (
            np.concatenate(logits, axis=0),
            np.concatenate(labels, axis=0),
            np.concatenate(entity_type, axis=0),
        )

        metrics = {}
        for entity_idx in range(self.model.num_entity_types):
            # in the entire batch, these indices contain the samples
            # corresponding to this entity type
            indices = np.argwhere(entity_type == entity_idx)

            # logits.shape = (batch_size, num_entity_types, seq_len,
            # num_features)
            preds = logits[indices, entity_idx, :, :]
            # (batch_size, seq_len, num_features)
            preds = np.squeeze(preds, axis=1)

            # (batch_size, seq_len)
            lab = labels[indices, :]
            lab = np.squeeze(lab, axis=1)

            if len(lab) == 0:
                preds, lab = [[]], [[]]
            metrics[str(entity_idx)] = compute_metrics((preds, lab))

        print(metrics)
        print("cellline: ", metrics["0"])
        print("tissue: ", metrics["1"])
        print("strain: ", metrics["2"])
        return metrics


if __name__ == "__main__":

    train_sets: List[Dataset] = [
        json_to_torch_dataset(path, tokenizer)
        for path in [
            "data/train_cell_line.json",
            "data/train_tissue.json",
            "data/train_strain.json",
        ]
    ]
    test_sets: List[Dataset] = [
        json_to_torch_dataset(path, tokenizer)
        for path in [
            "data/test_cell_line.json",
            "data/test_tissue.json",
            "data/test_strain.json",
        ]
    ]

    train_dataset = MultiNerDataset(train_sets)
    eval_dataset = MultiNerDataset(test_sets)

    collator = DataCollatorForTokenClassification(
        tokenizer, max_length=512, label_pad_token_id=-100, return_tensors="pt"
    )

    args = TrainingArguments(
        "artifacts/mtner",
        overwrite_output_dir=True,
        evaluation_strategy="steps",
        eval_steps=100,
        logging_first_step=True,
        save_strategy="epoch",
        disable_tqdm=True,
        learning_rate=2e-5,
        per_device_train_batch_size=20,
        per_device_eval_batch_size=20,
        num_train_epochs=8,
        weight_decay=0.01,
    )

    mtner = MultiNER(3)

    trainer = MultiNERTrainer(
        model=mtner,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )
    trainer.train()
