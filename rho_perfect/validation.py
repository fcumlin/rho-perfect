"""Validation of ρ-Perfect via split-raters and split-ratings (Section 3.1).

Both methods simulate two independent subjective assessments from a single one
and compare ρ-Perfect^2 (the predicted test-retest correlation) to the actual
Pearson correlation between the two simulated assessments (Eq. 10). See the
paper for details: Cumlin, F., "ρ-Perfect: Correlation Ceiling for Subjective
Evaluation Datasets", ICASSP 2026.
"""

import numpy as np
import pandas as pd
import scipy.stats

from rho_perfect import measure
from rho_perfect import utils


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------

def _aggregate_ratings_df(subjective_ratings: pd.DataFrame) -> pd.DataFrame:
    """Aggregate a per-rating DataFrame into mean, std, and n per item."""
    return (
        subjective_ratings.groupby("filename")["rating"]
        .agg(mean="mean", std=lambda x: x.std(ddof=1), n="count")
        .reset_index()
    )


def _compute_validation_iteration(
    subjective_ratings_a: pd.DataFrame,
    subjective_ratings_b: pd.DataFrame,
) -> dict:
    """Compute ρ-Perfect² on group A and Corr(Y1, Y2) between groups.

    Shared by both split-raters and split-ratings. ρ-Perfect is computed on
    group A only; the correlation is computed between the mean ratings of the
    two groups (Eq. 10).

    Args:
        subjective_ratings_a: Per-rating DataFrame for group A. Columns:
            filename, rating.
        subjective_ratings_b: Per-rating DataFrame for group B. Columns:
            filename, rating.

    Returns:
        Dictionary with keys rho_perfect, rho_perfect_squared, and corr_y1_y2.
    """
    aggregated_a = _aggregate_ratings_df(subjective_ratings_a)
    rho_perfect = measure.rho_perfect_from_aggregated(aggregated_a)

    # Corr(Y1, Y2): Pearson correlation between group means (Eq. 10).
    mean_a = subjective_ratings_a.groupby("filename")["rating"].mean()
    mean_b = subjective_ratings_b.groupby("filename")["rating"].mean()
    common_filenames = mean_a.index.intersection(mean_b.index)
    correlation, _ = scipy.stats.pearsonr(
        mean_a[common_filenames].values, mean_b[common_filenames].values
    )

    return {
        "rho_perfect": rho_perfect,
        "rho_perfect_squared": rho_perfect**2,
        "corr_y1_y2": correlation,
    }


# ---------------------------------------------------------------------------
# Validation functions.
# ---------------------------------------------------------------------------

def split_raters_validation(
    subjective_ratings: pd.DataFrame,
    n_iterations: int = 10,
    seed: int = 42,
) -> pd.DataFrame:
    """Validate ρ-Perfect using the split-raters method (Section 3.1).

    All unique raters are randomly partitioned into two equally sized groups.
    Ratings from each group form two independent assessments. ρ-Perfect² is
    computed on the first group and compared to the actual Pearson correlation
    between the two groups' mean ratings (Eq. 10). Raters that only appear on
    a subset of items are handled naturally: each item's means are computed
    from whichever raters in that group rated it.

    Args:
        subjective_ratings: Per-rating DataFrame. Columns: filename, rater_id,
            rating. Filename is the item identifier, rater_id identifies the
            rater, and rating is the individual subjective rating.
        n_iterations: Number of random splits to perform.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with one row per iteration and columns: iteration,
        rho_perfect, rho_perfect_sq (predicted test-retest correlation), and
        corr_y1_y2 (actual Pearson correlation between the two groups' means).
    """
    utils.validate_ratings_df(subjective_ratings)
    rng = np.random.default_rng(seed)
    unique_raters = subjective_ratings["rater_id"].unique()

    records = []
    for iteration in range(n_iterations):
        # Randomly partition all raters into two equal-sized groups; drop one
        # rater if the total count is odd.
        shuffled_raters = rng.permutation(unique_raters)
        half = len(shuffled_raters) // 2
        raters_a = set(shuffled_raters[:half])
        raters_b = set(shuffled_raters[half:2 * half])

        ratings_a = subjective_ratings[
            subjective_ratings["rater_id"].isin(raters_a)
        ][["filename", "rating"]]
        ratings_b = subjective_ratings[
            subjective_ratings["rater_id"].isin(raters_b)
        ][["filename", "rating"]]

        record = _compute_validation_iteration(ratings_a, ratings_b)
        records.append({"iteration": iteration, **record})

    return pd.DataFrame(records)


def split_ratings_validation(
    subjective_ratings: pd.DataFrame,
    n_iterations: int = 10,
    seed: int = 42,
) -> pd.DataFrame:
    """Validate ρ-Perfect using the split-ratings method (Section 3.1).

    For each item the individual ratings are randomly split into two equally
    sized groups, guaranteeing that every item has the same number of ratings
    in each group. Items with an odd number of ratings have one rating dropped
    before splitting. Note that a rater may appear in both groups across
    different items; this is expected behaviour of the method.

    ρ-Perfect^2 is computed on the first group and compared to the actual
    Pearson correlation between the two groups' mean ratings (Eq. 10).

    Args:
        subjective_ratings: Per-rating DataFrame. Columns: filename, rater_id,
            rating. Filename is the item identifier, rater_id identifies the
            rater, and rating is the individual subjective rating.
        n_iterations: Number of random splits to perform.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with one row per iteration and columns: iteration,
        rho_perfect, rho_perfect_sq (predicted test-retest correlation), and
        corr_y1_y2 (actual Pearson correlation between the two groups' means).
    """
    utils.validate_ratings_df(subjective_ratings)
    rng = np.random.default_rng(seed)

    records = []
    for iteration in range(n_iterations):
        rows_a = []
        rows_b = []

        for filename, item_df in subjective_ratings.groupby("filename"):
            ratings = item_df["rating"].to_numpy(dtype=float)

            # Drop one rating if odd so both halves are the same size.
            if len(ratings) % 2 != 0:
                ratings = ratings[:-1]

            permuted_indices = rng.permutation(len(ratings))
            half = len(ratings) // 2
            rows_a.append(pd.DataFrame({
                "filename": filename,
                "rating": ratings[permuted_indices[:half]],
            }))
            rows_b.append(pd.DataFrame({
                "filename": filename,
                "rating": ratings[permuted_indices[half:]],
            }))

        ratings_a = pd.concat(rows_a, ignore_index=True)
        ratings_b = pd.concat(rows_b, ignore_index=True)

        record = _compute_validation_iteration(ratings_a, ratings_b)
        records.append({"iteration": iteration, **record})

    return pd.DataFrame(records)