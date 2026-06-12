# mhqa/data.py
"""
Data loading, cleaning, deduplication, prompt construction, and tokenisation.
All decisions are grounded in EDA findings from Train.csv:
  - Columns: ID, input, output, subset
  - 1 empty-input row  → dropped
  - 467 intra-subset duplicate inputs → dropped
  - 1,002 cross-subset duplicate inputs → retained (different expected outputs)
  - MAX_INPUT_LEN=128 (100% of inputs fit), MAX_TARGET_LEN=384 (99%+ answers fit)
"""

import re
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from datasets import Dataset

from mhqa.constants import (
    ID_COL, QUESTION_COL, ANSWER_COL, SUBSET_COL, SUBSET_ORDER
)


# ── Text utilities ─────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Strip control characters and collapse whitespace. Safe for all scripts."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_prompt(question: str, subset: str) -> str:
    """
    Subset-prefixed prompt: 'Aka_Gha: <question>'
    The subset tag encodes both language and country, giving the model
    an explicit multilingual conditioning signal across all 8 configurations.
    """
    return f"{subset}: {clean_text(question)}"


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_split(path: str | Path, has_answer: bool = True) -> pd.DataFrame:
    """
    Load a single CSV split (Train, Val, or Test).
    Applies clean_text to input and output columns.
    """
    df = pd.read_csv(path)
    df[QUESTION_COL] = df[QUESTION_COL].apply(clean_text)
    if has_answer and ANSWER_COL in df.columns:
        df[ANSWER_COL] = df[ANSWER_COL].apply(clean_text)
    df["model_input"] = df.apply(
        lambda r: build_prompt(r[QUESTION_COL], r[SUBSET_COL]), axis=1
    )
    return df


def load_all(data_dir: str | Path) -> Dict[str, pd.DataFrame]:
    """
    Load train, val, and test splits from data_dir.
    Applies cleaning, deduplication (train only), and prompt construction.

    Returns:
        dict with keys 'train', 'val', 'test'
    """
    data_dir = Path(data_dir)
    train = load_split(data_dir / "Train.csv", has_answer=True)
    val   = load_split(data_dir / "Val.csv",   has_answer=True)
    test  = load_split(data_dir / "Test.csv",  has_answer=False)

    train = _deduplicate(train)
    return {"train": train, "val": val, "test": test}


def _deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """
    1. Drop 1 known empty-input row (ID_TR_Eng_Uga_E9A002A4).
    2. Drop intra-subset duplicate inputs (same question + same subset).
       Cross-subset duplicates (same question, different subset) are retained
       because they carry different expected outputs.
    """
    before = len(df)
    df = df[df[QUESTION_COL].str.len() > 0].copy()
    dropped_empty = before - len(df)

    before = len(df)
    df = df.drop_duplicates(subset=[QUESTION_COL, SUBSET_COL], keep="first")
    dropped_dups = before - len(df)

    df = df.reset_index(drop=True)
    print(f"  Deduplication: dropped {dropped_empty} empty row(s), "
          f"{dropped_dups} intra-subset duplicate(s). "
          f"Final train size: {len(df):,}")
    return df


# ── Stratified split ───────────────────────────────────────────────────────────

def stratified_split(
    df: pd.DataFrame,
    val_size: float = 0.05,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Stratified split by subset column so each language is represented
    proportionally in both the main and holdout sets.
    """
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
    train_idx, val_idx = next(splitter.split(df, df[SUBSET_COL]))
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[val_idx].reset_index(drop=True)


# ── HuggingFace Dataset construction ─────────────────────────────────────────

def make_hf_datasets(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    tokenizer,
    max_input_len: int = 128,
    max_target_len: int = 384,
):
    """
    Tokenise train and val DataFrames into HuggingFace Datasets ready for
    Seq2SeqTrainer. Returns (tok_train, tok_val).
    """
    def _tokenize(examples):
        model_inputs = tokenizer(
            examples["model_input"],
            max_length=max_input_len,
            truncation=True,
            padding=False,
        )
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                examples[ANSWER_COL],
                max_length=max_target_len,
                truncation=True,
                padding=False,
            )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    cols = ["model_input", ANSWER_COL]
    train_hf = Dataset.from_pandas(train_df[cols].reset_index(drop=True))
    val_hf   = Dataset.from_pandas(val_df[cols].reset_index(drop=True))

    tok_train = train_hf.map(_tokenize, batched=True, remove_columns=train_hf.column_names)
    tok_val   = val_hf.map(_tokenize,   batched=True, remove_columns=val_hf.column_names)

    return tok_train, tok_val
