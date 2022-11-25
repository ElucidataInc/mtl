import torch
from torch import nn
import numpy as np
from transformers import (
    AutoTokenizer,
    BertModel,
    BertConfig,
    TokenClassificationPipeline,
)
from transformers.pipelines import AggregationStrategy

pmbert_path = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"

device_id = 0 if torch.cuda.is_available() else -1


def create_mtner_pipeline(path):
    tokenizer = AutoTokenizer.from_pretrained(
        pmbert_path, model_max_length=512
    )
    model = MultiNER.from_pretrained(3, path)
    pipe = MtnerPipeline(
        model=model, tokenizer=tokenizer, framework="pt", device=device_id
    )

    def fn(text):
        return _mtner_wrapper(text, pipe)

    return fn


def _mtner_wrapper(text, pipeline):
    out = pipeline(text)
    res = []
    for idx, etype in enumerate(["cell_line", "tissue", "strain"]):
        res += _ner_wrapper(text, out[idx], pipeline, etype)
    return res


def _ner_wrapper(text, out, pipeline, etype):
    """
    Converts pubmedbert pipeline output to list of spans that look like this
    [
        {
            "score": 0.99932355,
            "start": 20,
            "end": 25,
            "label": "CELLLINE",
            "keyword": "MCF-7",
        },
        {
            "score": 0.98811394,
            "start": 92,
            "end": 98,
            "label": "CELLLINE",
            "keyword": "Hep-G2",
        },
    ]
    """
    tag_map = {
        "LABEL_0": "O",
        "LABEL_1": f"B-{etype}",
        "LABEL_2": f"I-{etype}",
        "O": "O",
        "B": f"B-{etype}",
        "I": f"I-{etype}",
    }
    out_processed = []
    for e in out:
        temp = e.copy()
        temp["entity"] = tag_map[e["entity"]]
        out_processed.append(temp)

    grouped_entities = pipeline.group_entities(out_processed)
    for ent in grouped_entities:
        ent["label"] = ent["entity_group"]
        del ent["entity_group"]

        ent["keyword"] = text[ent["start"] : ent["end"]]
        del ent["word"]

    grouped_entities = [ent for ent in grouped_entities if ent["label"] != "O"]
    return grouped_entities


class MultiNER(nn.Module):
    def __init__(self, num_entity_types: int):
        super().__init__()
        self.num_entity_types = num_entity_types
        # self.base_bert = pmbert
        self.base_bert = BertModel(BertConfig())
        self.dropout = nn.Dropout()
        classifiers = []
        for i in range(self.num_entity_types):
            classifiers.append(
                nn.Sequential(
                    nn.Linear(768, 768), nn.ReLU(), nn.Linear(768, 3)
                )
            )
        self.classifiers = nn.ModuleList(classifiers)
        # for param in self.base_bert.parameters():
        #     param.requires_grad = False
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


class MtnerPipeline(TokenClassificationPipeline):
    """
    This code is taken from TokenClassificationPipeline
    There are minor modifications
    """

    def _forward(self, model_inputs):
        # Forward
        special_tokens_mask = model_inputs.pop("special_tokens_mask")
        offset_mapping = model_inputs.pop("offset_mapping", None)
        sentence = model_inputs.pop("sentence")
        model_inputs.pop("token_type_ids")

        if self.framework == "tf":
            raise NotImplementedError()
        else:
            logits = self.model(**model_inputs)["logits"]

        return {
            "logits": logits,
            "special_tokens_mask": special_tokens_mask,
            "offset_mapping": offset_mapping,
            "sentence": sentence,
            **model_inputs,
        }

    def postprocess(
        self,
        model_outputs,
        aggregation_strategy=AggregationStrategy.NONE,
        ignore_labels=None,
    ):
        if ignore_labels is None:
            ignore_labels = ["O"]
        logits = model_outputs["logits"][0].numpy()
        sentence = model_outputs["sentence"]
        input_ids = model_outputs["input_ids"][0]
        offset_mapping = (
            model_outputs["offset_mapping"][0]
            if model_outputs["offset_mapping"] is not None
            else None
        )
        special_tokens_mask = model_outputs["special_tokens_mask"][0].numpy()

        maxes = np.max(logits, axis=-1, keepdims=True)
        shifted_exp = np.exp(logits - maxes)
        scores = shifted_exp / shifted_exp.sum(axis=-1, keepdims=True)

        entities = {}
        for entity_idx in range(self.model.num_entity_types):
            pre_entities = self.gather_pre_entities(
                sentence,
                input_ids,
                scores[entity_idx],
                offset_mapping,
                special_tokens_mask,
                aggregation_strategy,
            )
            grouped_entities = self.aggregate(
                pre_entities, aggregation_strategy
            )
            # Filter anything that is in self.ignore_labels
            entities[entity_idx] = [
                entity
                for entity in grouped_entities
                if entity.get("entity", None) not in ignore_labels
                and entity.get("entity_group", None) not in ignore_labels
            ]
        return entities
