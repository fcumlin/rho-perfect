"""Estimates correlation ceiling (ρ-Perfect) for subjectively rated datasets.

The ρ-Perfect metric estimates the maximum correlation that any predictive model
can achieve on a dataset of subjective ratings, given the inherent noise in
human ratings. The implementation is the official implementation of ρ-Perfect,
presented in the following paper: Cumlin, F., "ρ-Perfect: Correlation Ceiling
for Subjective Evaluation Datasets", ICASSP 2026. All equations refered to in
the comments are from that paper.

Two input formats are supported:
    1. Aggregated: DataFrame with columns: filename, mean, std, n (mean = mean
        rating, std = standard deviation of ratings, n = number of ratings)
    2. Per-rating: DataFrame with columns: filename, rating (rating = individual
        ratings).
"""
import warnings

import numpy as np
import pandas as pd
import scipy.stats

from rho_perfect import utils


def calculate_rho_perfect(
    subjective_statistics: pd.DataFrame,
    ddof: int = 1
) -> float:
    """Calculates ρ-Perfect from an aggregated DataFrame.

    Args:
        subjective_statistics: pd.DataFrame with columns: filename, mean, std,
            and n. Filename is the item identifier, mean is the average rating
            value on the filename, std is the standard deviation of the ratings
            on the item, and n is the number of ratings on the item. Each row is
            one item. Recommended is n >= 3 for every item and at least 50
            items.
        ddof: 0 if population standard deviation is used, 1 if sample standard
            deviation is used to compute std column in subjective_statistics.
            Compare to np.std's ddof parameter.

    Returns:
        The ρ-Perfect value (correlation ceiling).
    """
    utils.validate_aggregated_df(subjective_statistics)

    mean = subjective_statistics["mean"].to_numpy(dtype=float)
    std = subjective_statistics["std"].to_numpy(dtype=float)
    if ddof == 0:
        std = std * np.sqrt(
            subjective_statistics["n"] / (subjective_statistics["n"] - 1)
        )
    n = subjective_statistics["n"].to_numpy(dtype=float)

    # Var(Y): Variance of the mean ratings across items (Eq. 3).
    var_y = np.var(mean, ddof=1)

    # E[Var(Y|X)]: Average within-item variance of the mean (Eq. 5 & 6).
    var_y_given_x = np.mean(std**2 / n)

    # Var(Y_hat) = Var(Y) - E[Var(Y|X)] (Eq. 4 rearranged).
    var_y_hat = var_y - var_y_given_x

    # In the unlikely but possible case that the variance of the observed
    # ratings is zero, the ceiling is zero, since no model can correlate with a
    # constant target.
    if var_y == 0:
        warnings.warn(
            "Variance of mean ratings across items is zero. The target is "
            "constant; ρ-Perfect is not meaningful for this dataset. Returning "
            "0."
        )
        return 0.0

    if var_y_hat <= 0:
        raise ValueError(
            "Estimated Var(Ŷ) is non-positive. The noise dominates the signal; "
            "ρ-Perfect is not meaningful for this dataset."
        )

    return float(np.sqrt(var_y_hat / var_y))


def calculate_rho_perfect_from_ratings(
    subjective_ratings: pd.DataFrame
) -> float:
    """Compute ρ-Perfect from a per-rating DataFrame.

    Args:
        subjective_ratings: pd.DataFrame with columns: filename, and rating.
            Filename is the item identifier, and rating is the individual
            subjective rating.

    Returns:
        The ρ-Perfect value (correlation ceiling).
    """
    utils.validate_ratings_df(subjective_ratings)
    agg = (
        subjective_ratings.groupby("filename")["rating"]
        .agg(mean="mean", std=lambda x: x.std(ddof=1), n="count")
        .reset_index()
    )
    return calculate_rho_perfect(agg)