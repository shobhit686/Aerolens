"""
AeroLens â€“ Data Preprocessing Module
=====================================
Cleans raw customer-feedback data, normalises text, removes duplicates,
handles missing values, and saves the processed dataset.

Outputs:
  data/processed/feedback_cleaned.csv   â€“ full cleaned dataset
  data/processed/train.csv              â€“ 80 % stratified training split
  data/processed/test.csv               â€“ 20 % stratified test split
"""

import os
import re
import string

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# Text-cleaning utilities
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """
    Normalise a single review string:
      1. Lowercase
      2. Remove URLs
      3. Remove email addresses
      4. Remove special characters (keep letters, digits, spaces)
      5. Collapse multiple spaces
      6. Strip leading / trailing whitespace
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", "", text)           # URLs
    text = re.sub(r"\S+@\S+", "", text)                     # emails
    text = re.sub(r"[^a-z0-9\s]", " ", text)                # non-alphanum
    text = re.sub(r"\s+", " ", text).strip()                 # extra spaces
    return text


# ---------------------------------------------------------------------------
# Main preprocessing pipeline
# ---------------------------------------------------------------------------

def preprocess(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Full preprocessing pipeline:
      1. Drop exact duplicate rows.
      2. Drop rows with missing review_text.
      3. Fill missing satisfaction_rating with the column median.
      4. Clean and normalise review_text â†’ cleaned_text column.
      5. Remove any rows where cleaned_text is empty after cleaning.

    Returns the cleaned DataFrame (original columns + cleaned_text).
    """
    n_start = len(df)

    # 1. Drop duplicates
    df = df.drop_duplicates(subset=["review_text", "airline", "feedback_date"], keep="first")
    n_dedup = n_start - len(df)

    # 2. Drop rows without review text
    df = df.dropna(subset=["review_text"])
    df = df[df["review_text"].str.strip() != ""]

    # 3. Handle missing satisfaction ratings (fill with median)
    if "satisfaction_rating" in df.columns:
        median_rating = df["satisfaction_rating"].median()
        df["satisfaction_rating"] = df["satisfaction_rating"].fillna(median_rating)

    # 4. Clean text
    df = df.copy()
    df["cleaned_text"] = df["review_text"].apply(clean_text)

    # 5. Remove empty cleaned rows
    df = df[df["cleaned_text"].str.len() > 0]

    df = df.reset_index(drop=True)

    if verbose:
        print(f"  Duplicates removed   : {n_dedup}")
        print(f"  Records after cleaning: {len(df)}")

    return df


def create_train_test_split(
    df: pd.DataFrame,
    target_col: str = "sentiment",
    test_size: float = 0.20,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform a stratified 80:20 train-test split.
    Stratification ensures each sentiment class is proportionally represented
    in both splits, preventing skewed evaluation.
    """
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=df[target_col],
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    return train_df, test_df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    raw_path = os.path.join(project_root, "data", "raw", "feedback_raw.csv")
    processed_dir = os.path.join(project_root, "data", "processed")
    os.makedirs(processed_dir, exist_ok=True)

    print("=" * 60)
    print("AeroLens â€“ Data Preprocessing")
    print("=" * 60)

    # Load raw data
    if not os.path.exists(raw_path):
        print(f"\nâœ— Raw data not found at {raw_path}")
        print("  Run generate_data.py first.\n")
        raise SystemExit(1)

    df_raw = pd.read_csv(raw_path)
    print(f"\n  Raw records loaded   : {len(df_raw)}")

    # Preprocess
    df_clean = preprocess(df_raw, verbose=True)

    # Save cleaned dataset
    cleaned_path = os.path.join(processed_dir, "feedback_cleaned.csv")
    df_clean.to_csv(cleaned_path, index=False)
    print(f"\nâœ“ Cleaned data saved to: {cleaned_path}")

    # Train / test split (stratified on sentiment)
    train_df, test_df = create_train_test_split(df_clean, target_col="sentiment")

    train_path = os.path.join(processed_dir, "train.csv")
    test_path = os.path.join(processed_dir, "test.csv")
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"âœ“ Training set : {len(train_df)} records â†’ {train_path}")
    print(f"âœ“ Test set     : {len(test_df)} records â†’ {test_path}")

    print(f"\nSentiment split (train):")
    print(train_df["sentiment"].value_counts().to_string())
    print(f"\nSentiment split (test):")
    print(test_df["sentiment"].value_counts().to_string())
    print()

