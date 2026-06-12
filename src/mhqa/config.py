# mhqa/config.py
"""
Experiment configuration as a typed dataclass.
load_config() reads a YAML file and returns a TrainingConfig.
select_config() auto-selects base vs large based on available VRAM.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import yaml


@dataclass
class TrainingConfig:
    # Model
    model_name: str = "google/mt5-base"
    experiment_name: str = "exp01_baseline"

    # Tokenisation — justified by EDA:
    #   100% of inputs fit in 128 tokens; 99%+ of answers fit in 384 tokens
    max_input_length: int = 128
    max_target_length: int = 384

    # Training
    num_train_epochs: int = 5
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 4   # effective batch = 32
    learning_rate: float = 3e-4
    warmup_ratio: float = 0.06
    weight_decay: float = 0.01
    lr_scheduler_type: str = "cosine"
    label_smoothing_factor: float = 0.1
    early_stopping_patience: int = 2
    seed: int = 42

    # PEFT / LoRA
    use_peft: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list = field(default_factory=lambda: ["q", "v"])

    # Inference
    num_beams: int = 4
    no_repeat_ngram_size: int = 3

    # Paths
    data_dir: str = "data"
    output_dir: str = "outputs"
    submission_dir: str = "outputs/submissions"
    reports_dir: str = "reports"

    # Logging
    logging_steps: int = 100
    report_to: str = "none"        # set "wandb" to enable Weights & Biases
    dataloader_num_workers: int = 0  # 0 = safe on Windows; increase on Linux/Colab

    @property
    def train_path(self) -> str:
        return str(Path(self.data_dir) / "Train.csv")

    @property
    def val_path(self) -> str:
        return str(Path(self.data_dir) / "Val.csv")

    @property
    def test_path(self) -> str:
        return str(Path(self.data_dir) / "Test.csv")

    @property
    def sample_submission_path(self) -> str:
        return str(Path(self.data_dir) / "SampleSubmission.csv")

    @property
    def submission_path(self) -> str:
        return str(Path(self.submission_dir) / f"submission_{self.experiment_name}.csv")

    @property
    def checkpoint_dir(self) -> str:
        return str(Path(self.output_dir) / "checkpoints" / self.experiment_name)

    @property
    def best_model_dir(self) -> str:
        return str(Path(self.output_dir) / "best_model" / self.experiment_name)

    def effective_batch_size(self) -> int:
        return self.per_device_train_batch_size * self.gradient_accumulation_steps

    def summary(self) -> str:
        lines = [
            f"Experiment      : {self.experiment_name}",
            f"Model           : {self.model_name}",
            f"PEFT            : {'LoRA r=%d α=%d' % (self.lora_r, self.lora_alpha) if self.use_peft else 'Full fine-tune'}",
            f"LR              : {self.learning_rate}",
            f"Effective batch : {self.effective_batch_size()}",
            f"Epochs          : {self.num_train_epochs}",
            f"MAX_INPUT       : {self.max_input_length}",
            f"MAX_TARGET      : {self.max_target_length}",
            f"Num beams       : {self.num_beams}",
            f"Label smoothing : {self.label_smoothing_factor}",
        ]
        return "\n".join(lines)


def load_config(path: str | Path) -> TrainingConfig:
    """Load a YAML config file and return a TrainingConfig dataclass."""
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return TrainingConfig(**{k: v for k, v in data.items()
                             if k in TrainingConfig.__dataclass_fields__})


def select_config(configs_dir: str | Path = "configs") -> tuple[TrainingConfig, str]:
    """
    Auto-select mt5_base or mt5_large based on available GPU VRAM.
    Returns (config, config_path_str).
    """
    import torch
    configs_dir = Path(configs_dir)
    has_gpu = torch.cuda.is_available()
    vram_gb = (torch.cuda.get_device_properties(0).total_memory / 1e9
               if has_gpu else 0)

    if has_gpu and vram_gb > 15:
        path = configs_dir / "mt5_large.yaml"
        reason = f"{vram_gb:.1f} GB VRAM — large model fits"
    else:
        path = configs_dir / "mt5_base.yaml"
        reason = f"{'%.1f GB VRAM' % vram_gb if has_gpu else 'CPU'} — using safe default"

    print(f"Auto-selected: {path.name}  ({reason})")
    return load_config(path), str(path)
