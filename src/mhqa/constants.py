# mhqa/constants.py
# ─────────────────────────────────────────────────────────────────────────────
# Single source of truth for column names, subset metadata, and colour palette.
# Verified against Train.csv (columns: ID, input, output, subset).

ID_COL       = "ID"
QUESTION_COL = "input"
ANSWER_COL   = "output"
SUBSET_COL   = "subset"

SUBSET_ORDER = [
    "Aka_Gha", "Amh_Eth", "Eng_Eth", "Eng_Gha",
    "Eng_Ken", "Eng_Uga", "Lug_Uga", "Swa_Ken",
]

SUBSET_LABELS = {
    "Aka_Gha": "Akan (Ghana)",
    "Amh_Eth": "Amharic (Ethiopia)",
    "Eng_Eth": "English (Ethiopia)",
    "Eng_Gha": "English (Ghana)",
    "Eng_Ken": "English (Kenya)",
    "Eng_Uga": "English (Uganda)",
    "Lug_Uga": "Luganda (Uganda)",
    "Swa_Ken": "Swahili (Kenya)",
}

PALETTE = {
    "Aka_Gha": "#2D6A9F",
    "Amh_Eth": "#E05C2A",
    "Eng_Eth": "#E8A838",
    "Eng_Gha": "#F0C040",
    "Eng_Ken": "#7DB8D8",
    "Eng_Uga": "#A8C8E8",
    "Lug_Uga": "#4CAF82",
    "Swa_Ken": "#8E5EA2",
}

SHORT_LABELS = [
    "Akan\n(GHA)", "Amh\n(ETH)", "Eng\n(ETH)", "Eng\n(GHA)",
    "Eng\n(KEN)", "Eng\n(UGA)", "Lug\n(UGA)", "Swa\n(KEN)",
]

LANG_FAMILY_MAP = {
    "Aka_Gha": "Akan",
    "Amh_Eth": "Amharic",
    "Eng_Eth": "English",
    "Eng_Gha": "English",
    "Eng_Ken": "English",
    "Eng_Uga": "English",
    "Lug_Uga": "Luganda",
    "Swa_Ken": "Swahili",
}
