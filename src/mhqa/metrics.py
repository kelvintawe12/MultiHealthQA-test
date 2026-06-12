# mhqa/metrics.py
"""
ROUGE evaluation helpers.
  - compute_rouge()             : overall ROUGE-1, ROUGE-2, ROUGE-L
  - compute_rouge_by_language() : per-subset breakdown as a DataFrame
  - make_compute_metrics()      : factory for HuggingFace Trainer
"""

from typing import List
import numpy as np
import pandas as pd
import evaluate as hf_evaluate

from mhqa.constants import SUBSET_ORDER, SUBSET_LABELS

_rouge = hf_evaluate.load("rouge")


def compute_rouge(
    predictions: List[str],
    references: List[str],
    use_stemmer: bool = True,
) -> dict:
    """
    Compute ROUGE-1, ROUGE-2, ROUGE-L on two aligned lists.
    Returns a dict with keys rouge1, rouge2, rougeL (F1 scores, 0-1).
    """
    result = _rouge.compute(
        predictions=predictions,
        references=references,
        use_stemmer=use_stemmer,
    )
    return {k: round(v, 4) for k, v in result.items()}


def compute_rouge_by_language(
    predictions: List[str],
    references: List[str],
    subsets: List[str],
    use_stemmer: bool = True,
) -> pd.DataFrame:
    """
    Compute ROUGE per subset and return a tidy DataFrame.
    Rows are ordered by SUBSET_ORDER. An 'Overall' row is appended.
    """
    rows = []
    for subset in SUBSET_ORDER:
        idx = [i for i, s in enumerate(subsets) if s == subset]
        if not idx:
            continue
        preds = [predictions[i] for i in idx]
        refs  = [references[i] for i in idx]
        res   = compute_rouge(preds, refs, use_stemmer)
        rows.append({
            "Subset":  SUBSET_LABELS[subset],
            "n":       len(idx),
            "ROUGE-1": res["rouge1"],
            "ROUGE-2": res["rouge2"],
            "ROUGE-L": res["rougeL"],
        })

    # Overall row
    overall = compute_rouge(predictions, references, use_stemmer)
    rows.append({
        "Subset":  "Overall",
        "n":       len(predictions),
        "ROUGE-1": overall["rouge1"],
        "ROUGE-2": overall["rouge2"],
        "ROUGE-L": overall["rougeL"],
    })

    return pd.DataFrame(rows).set_index("Subset")


def make_compute_metrics(tokenizer, use_stemmer: bool = True):
    """
    Returns a compute_metrics function compatible with HuggingFace Seq2SeqTrainer.
    """
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_preds  = [p.strip() for p in tokenizer.batch_decode(preds,  skip_special_tokens=True)]
        decoded_labels = [l.strip() for l in tokenizer.batch_decode(labels, skip_special_tokens=True)]
        result = _rouge.compute(
            predictions=decoded_preds,
            references=decoded_labels,
            use_stemmer=use_stemmer,
        )
        return {k: round(v, 4) for k, v in result.items()}

    return compute_metrics
