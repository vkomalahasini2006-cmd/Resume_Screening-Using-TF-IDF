import os
from typing import Dict, List, Optional, Tuple

import pandas as pd

DEFAULT_TEXT_COLUMNS = ["Resume_str", "Resume_str ", "Resume", "resume_str", "resume"]
DEFAULT_ID_COLUMNS = ["ID", "Id", "id", "ResumeID", "resume_id", "filename"]
DEFAULT_CATEGORY_COLUMNS = ["Category", "category", "Label", "label"]


def _find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for column in candidates:
        if column in df.columns:
            return column
    return None


def load_resume_dataset_csv(
    csv_path: str,
    text_column: Optional[str] = None,
    id_column: Optional[str] = None,
    category_column: Optional[str] = None,
    max_samples: Optional[int] = None,
) -> Tuple[Dict[str, str], pd.DataFrame]:
    """Load a Kaggle resume dataset CSV and return resume text mapping plus the DataFrame."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"Dataset CSV is empty: {csv_path}")

    text_column = text_column or _find_column(df, DEFAULT_TEXT_COLUMNS)
    if text_column is None:
        raise ValueError(
            "Could not find a resume text column. Expected one of: "
            + ", ".join(DEFAULT_TEXT_COLUMNS)
        )

    id_column = id_column or _find_column(df, DEFAULT_ID_COLUMNS)
    if id_column is None:
        raise ValueError(
            "Could not find an ID column. Expected one of: "
            + ", ".join(DEFAULT_ID_COLUMNS)
        )

    category_column = category_column or _find_column(df, DEFAULT_CATEGORY_COLUMNS)

    df[text_column] = df[text_column].astype(str).fillna("")
    df[id_column] = df[id_column].astype(str).fillna("")

    df = df[df[text_column].str.strip() != ""].copy()
    if df.empty:
        raise ValueError(f"No non-empty resume text rows found in {csv_path}")

    if max_samples is not None and max_samples > 0:
        df = df.head(max_samples)

    resume_texts = {
        row[id_column]: row[text_column]
        for _, row in df.iterrows()
        if row[id_column].strip() != ""
    }

    if not resume_texts:
        raise ValueError("No valid resume entries found after loading the dataset.")

    return resume_texts, df


def load_kaggle_resume_dataset(
    csv_path: str,
    max_samples: Optional[int] = None,
) -> Tuple[Dict[str, str], pd.DataFrame]:
    """Convenience loader for the Kaggle resume dataset structure."""
    return load_resume_dataset_csv(
        csv_path,
        text_column="Resume_str",
        id_column="ID",
        category_column="Category",
        max_samples=max_samples,
    )
