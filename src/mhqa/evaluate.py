# mhqa/evaluate.py
"""
Evaluation helpers used in the notebook.
  - holdout_for()    : creates a stratified holdout from the validation set
  - evaluate_model() : runs inference + ROUGE on a holdout DataFrame
"""

from typing import Tuple
import pandas as pd

from mhqa.config import TrainingConfig
from mhqa.data import load_split, stratified_split
from mhqa.infer import predict_dataframe
from mhqa.metrics import compute_rouge, compute_rouge_by_language
from mhqa.constants import ANSWER_COL, SUBSET_COL


def holdout_for(
    cfg: TrainingConfig,
    holdout_fraction: float = 0.10,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Load the validation CSV and return a stratified holdout sample.
    Default: 10% of val set, stratified by subset.
    """
    val_df = load_split(cfg.val_path, has_answer=True)
    _, holdout = stratified_split(val_df, val_size=holdout_fraction, seed=seed)
    print(f"Holdout: {len(holdout)} rows  ({holdout_fraction*100:.0f}% of val set)")
    return holdout


def evaluate_model(
    model,
    tokenizer,
    holdout: pd.DataFrame,
    cfg: TrainingConfig,
    batch_size: int = 16,
    device: str = "cpu",
) -> Tuple[dict, pd.DataFrame, list]:
    """
    Run inference on holdout and compute ROUGE scores.

    Returns:
        overall  : dict with rouge1, rouge2, rougeL (F1 scores)
        per_lang : DataFrame with per-subset ROUGE breakdown
        preds    : raw prediction strings
    """
    preds = predict_dataframe(model, tokenizer, holdout, cfg,
                              batch_size=batch_size, device=device)

    refs    = holdout[ANSWER_COL].tolist()
    subsets = holdout[SUBSET_COL].tolist()

    overall  = compute_rouge(preds, refs)
    per_lang = compute_rouge_by_language(preds, refs, subsets)

    return overall, per_lang, preds
