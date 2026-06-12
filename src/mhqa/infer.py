# mhqa/infer.py
"""
Inference utilities.
  - generate_batch()     : raw list of texts → list of predictions
  - predict_dataframe()  : DataFrame with model_input column → list of predictions
"""

from typing import List
import torch
import pandas as pd
from tqdm.auto import tqdm

from mhqa.config import TrainingConfig


def generate_batch(
    texts: List[str],
    model,
    tokenizer,
    cfg: TrainingConfig,
    batch_size: int = 16,
    device: str = "cpu",
) -> List[str]:
    """
    Generate predictions for a list of input strings.
    Handles batching, device placement, and decoding.
    """
    model.eval()
    outputs = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Generating"):
        batch = texts[i: i + batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=cfg.max_input_length,
        ).to(device)

        with torch.no_grad():
            gen_ids = model.generate(
                **enc,
                max_new_tokens=cfg.max_target_length,
                num_beams=cfg.num_beams,
                no_repeat_ngram_size=cfg.no_repeat_ngram_size,
                early_stopping=True,
            )

        decoded = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        outputs.extend([d.strip() for d in decoded])

    return outputs


def predict_dataframe(
    model,
    tokenizer,
    df: pd.DataFrame,
    cfg: TrainingConfig,
    batch_size: int = 16,
    device: str = "cpu",
    retriever=None,
) -> List[str]:
    """
    Run inference on an entire DataFrame.
    If retriever is provided and cfg.hybrid_fallback is True,
    uses retrieved answer when model output is empty.
    """
    texts = df["model_input"].tolist()
    preds = generate_batch(texts, model, tokenizer, cfg, batch_size=batch_size, device=device)

    # Hybrid fallback: replace empty model outputs with retrieval
    if retriever is not None and getattr(cfg, "hybrid_fallback", False):
        ret_preds, _, _ = retriever.predict(df)
        preds = [
            p if p.strip() else r
            for p, r in zip(preds, ret_preds)
        ]

    return preds
