# Multilingual Health QA — Low-Resource African Languages
**Zindi Competition | Machine Learning Techniques I — Final Project**

---

## Project Structure

```
multilingual-health-qa/
├── Multilingual_Health_QA_Final.ipynb   ← Main notebook (run this)
├── requirements.txt
├── README.md
│
├── src/mhqa/                            ← Reusable package (no phantom imports)
│   ├── __init__.py
│   ├── constants.py    ← Column names, subset metadata, colour palette
│   ├── config.py       ← TrainingConfig dataclass + YAML loader
│   ├── data.py         ← load_all, load_split, stratified_split, make_hf_datasets
│   ├── modeling.py     ← load_tokenizer, load_model (torch_dtype fixed)
│   ├── metrics.py      ← compute_rouge, compute_rouge_by_language, make_compute_metrics
│   ├── retrieval.py    ← PerLanguageRetriever (TF-IDF baseline)
│   ├── infer.py        ← generate_batch, predict_dataframe
│   ├── evaluate.py     ← holdout_for, evaluate_model
│   ├── submit.py       ← make_submission (validates ID alignment)
│   └── eda.py          ← All figure-generation functions
│
├── configs/
│   ├── mt5_base.yaml   ← Default (safe for T4 GPU)
│   └── mt5_large.yaml  ← Requires ≥14 GB VRAM
│
├── data/               ← Place competition CSVs here
│   ├── Train.csv
│   ├── Val.csv
│   ├── Test.csv
│   └── SampleSubmission.csv
│
├── outputs/
│   ├── plots/          ← EDA and training figures
│   ├── checkpoints/    ← Model checkpoints per experiment
│   ├── best_model/     ← Saved model weights + tokeniser
│   └── submissions/    ← Generated submission CSVs
│
└── reports/
    ├── experiments.csv ← Auto-updated experiment log
    └── figures/        ← EDA figure copies
```

---

## Setup — VS Code (Local)

### Prerequisites
- Python 3.10 or 3.11
- VS Code + **Jupyter** extension
- GPU with CUDA 11.8+ (training on CPU is possible but very slow)

### Step 1 — Unzip and open
```bash
unzip multilingual_health_qa.zip
cd multilingual_health_qa
code .
```

### Step 2 — Virtual environment
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### Step 3 — Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **GPU users (CUDA 11.8):**
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cu118
> pip install -r requirements.txt
> ```

> **Windows (native, no WSL2):** Comment out `bitsandbytes` in `requirements.txt`.
> LoRA training still works — only 8-bit quantisation is disabled.

### Step 4 — Place data files
```
data/
├── Train.csv
├── Val.csv
├── Test.csv
└── SampleSubmission.csv
```

### Step 5 — Run the notebook
1. Open `Multilingual_Health_QA_Final.ipynb` in VS Code
2. **Select Kernel → Python Environments → `.venv`**
3. Run cells top to bottom with **Shift+Enter**

---

## Google Colab

1. Upload `Multilingual_Health_QA_Final.ipynb` to Colab
2. Runtime → Change runtime type → **GPU (T4)**
3. In Cell 0, uncomment the `!pip install ...` line and run it
4. Set `IS_COLAB = True` in Cell 1 and mount Drive
5. Place CSVs in `MyDrive/multilingual_health_qa/`
6. Run all cells

---

## Experiment Guide

Change **one variable per run** via the config override cell (Cell 4):

| ID | What changes | Config change |
|---|---|---|
| EXP-01 | Baseline: mt5-small, plain prompt | `cfg.model_name = "google/mt5-small"` |
| EXP-02 | mt5-base, plain prompt | default YAML |
| EXP-03 | Subset prefix prompt | default YAML |
| EXP-04 | LR 5e-4 → 3e-4 | `cfg.learning_rate = 3e-4` |
| EXP-05 | MAX_TARGET 256 → 384 | default YAML (already 384) |
| EXP-06 | LoRA rank r=16 → r=32 | `cfg.lora_r = 32; cfg.lora_alpha = 64` |
| EXP-07 | Deduplication active | default (already in `load_all`) |
| EXP-08 | Full fine-tune (no PEFT) | `cfg.use_peft = False; cfg.learning_rate = 1e-4` |
| EXP-09 | mt5-large | `cfg = load_config("configs/mt5_large.yaml")` |
| EXP-10 | label_smoothing + beam=8 | `cfg.num_beams = 8` |

---

## Bugs Fixed vs Previous Version

| Previous broken code | Fixed in this version |
|---|---|
| `from mhqa.data import load_all` — package did not exist | `src/mhqa/data.py` created and fully implemented |
| `torch_torch_dtype=...` typo in modeling.py | Corrected to `torch_dtype=` in `mhqa/modeling.py` |
| `from mhqa.retrieval import PerLanguageRetriever` — missing | Implemented in `mhqa/retrieval.py` |
| `from mhqa.evaluate import evaluate_model, holdout_for` — missing | Implemented in `mhqa/evaluate.py` |
| `from mhqa.submit import make_submission` — missing | Implemented in `mhqa/submit.py` |
| `from mhqa.config import load_config` — missing | Implemented in `mhqa/config.py` + YAML files |
| `cfg` / `trainer` / `holdout` used before definition | All variables defined in correct cell order |
| `scripts/run_eda`, `scripts/train` — missing scripts | All logic moved inline to `mhqa/eda.py` and notebook |
| No `configs/mt5_base.yaml` or `mt5_large.yaml` | Both files created |
| `os._exit(00)` kernel restart mid-notebook | Removed — was causing session loss |

---

*AI tools assisted with code scaffolding. All design decisions, experiments, and analysis are the author's own work, grounded in exploratory analysis of the actual Train.csv dataset.*
