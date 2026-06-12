# mhqa/submit.py
"""
Submission file builder.
make_submission() aligns predictions to the SampleSubmission ID order,
validates shape and detects any missing IDs before saving.
"""

from pathlib import Path
from typing import List

import pandas as pd

from mhqa.constants import ID_COL


def make_submission(
    ids: List[str],
    predictions: List[str],
    output_path: str,
    sample_submission_path: str,
) -> pd.DataFrame:
    """
    Build a submission CSV aligned to the SampleSubmission order.

    Args:
        ids                   : ID values from test_df (same order as predictions)
        predictions           : model output strings
        output_path           : where to save the CSV
        sample_submission_path: path to SampleSubmission.csv (validates structure)

    Returns:
        The saved DataFrame.

    Raises:
        ValueError if ID sets do not match between predictions and sample.
    """
    sample = pd.read_csv(sample_submission_path)
    answer_col = [c for c in sample.columns if c.lower() != "id"][0]

    # Build prediction map
    pred_map = dict(zip(ids, predictions))

    # Validate coverage
    sample_ids = set(sample[ID_COL])
    pred_ids   = set(ids)
    missing    = sample_ids - pred_ids
    extra      = pred_ids - sample_ids

    if missing:
        raise ValueError(
            f"make_submission: {len(missing)} IDs in SampleSubmission are missing "
            f"from your predictions. First 5: {list(missing)[:5]}"
        )
    if extra:
        print(f"  Warning: {len(extra)} extra prediction IDs not in SampleSubmission — ignored.")

    # Align to sample order
    submission = sample.copy()
    submission[answer_col] = submission[ID_COL].map(pred_map)

    n_null = submission[answer_col].isnull().sum()
    if n_null:
        print(f"  Warning: {n_null} rows have null predictions after alignment.")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    print(f"Submission saved → {output_path}  ({len(submission)} rows)")

    return submission
