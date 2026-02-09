"""Validation utilities for ρ-Perfect."""

import warnings

import pandas as pd


def validate_aggregated_df(df: pd.DataFrame) -> None:
    """Validate an aggregated DataFrame for required columns and sizes.

    Args:
        df: DataFrame that should contain columns: filename, mean, std, n.
            Raises ValueError if any are missing. Warns if fewer than 50 items
            or if any item has fewer than 3 ratings.
    """
    required = {"filename", "mean", "std", "n"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Aggregated DataFrame is missing columns: {missing}")
    if df['mean'].nunique() == 1:
        raise ValueError(
            "All item means are identical. Variance of Y is zero; correlation "
            "ceiling is not meaningful for this dataset.",
        )
    if (df["n"] == 1).any():
        raise ValueError(
            "Some items have only one rating. Remove those items or provide "
            "ratings for them to compute ρ-Perfect.",
        )
    if len(df) < 50:
        warnings.warn(
            f"Dataset has {len(df)} items; at least 50 are recommended. The "
            "estimate may be unreliable.",
            UserWarning,
            stacklevel=3,
        )
    if (df["n"] < 3).any():
        warnings.warn(
            "Some items have fewer than 3 ratings. Within-item variance "
            "estimates for those items will be unreliable.",
            UserWarning,
            stacklevel=3,
        )


def validate_ratings_df(df: pd.DataFrame) -> None:
    """Validate a per-rating DataFrame for required columns.

    Args:
        df: DataFrame that should contain columns: filename and rating.
            Raises ValueError if any are missing.
    """
    required = {"filename", "rating"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Per-rating DataFrame is missing columns: {missing}")