# mhqa/modeling.py
"""
Model and tokeniser loading with optional PEFT/LoRA.
Fixed: torch_dtype (was torch_torch_dtype in previous broken version).
"""

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import LoraConfig, get_peft_model, TaskType

from mhqa.config import TrainingConfig


def load_tokenizer(cfg: TrainingConfig):
    """Load the mT5 tokeniser for the configured model."""
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    print(f"Tokenizer loaded: {cfg.model_name}  (vocab size: {tokenizer.vocab_size:,})")
    return tokenizer


def load_model(cfg: TrainingConfig, device: str = "cpu"):
    """
    Load the seq2seq model with correct torch_dtype (float16 on GPU, float32 on CPU).
    Applies LoRA PEFT if cfg.use_peft is True.
    """
    dtype = torch.float16 if device == "cuda" else torch.float32

    model = AutoModelForSeq2SeqLM.from_pretrained(
        cfg.model_name,
        torch_dtype=dtype,                          # ← correct; was torch_torch_dtype
        device_map="auto" if device == "cuda" else None,
    )

    if cfg.use_peft:
        peft_config = LoraConfig(
            task_type      = TaskType.SEQ_2_SEQ_LM,
            r              = cfg.lora_r,
            lora_alpha     = cfg.lora_alpha,
            lora_dropout   = cfg.lora_dropout,
            target_modules = cfg.lora_target_modules,
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    else:
        total     = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Full fine-tune — total: {total:,} | trainable: {trainable:,}")

    return model
