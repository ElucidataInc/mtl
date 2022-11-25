## Data

Labeled data for tissue, strain and cell line NER is present in the `data/` directory.

## Training

`main.py` contains the code for training a multi-task model for these entities.

## Inference

Once the model is trained, the `create_mtner_pipeline` function in `inferencey.py` can be used to create a callable which returns entity spans

```python
pipe = create_mtner_pipeline("<model_path>")

pipe("liver biopsy")
# {'keyword': 'liver', 'entity_type': 'tissue', 'span_begin': 0, 'span_end': 5}
```
