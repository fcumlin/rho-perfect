"""Tests for rho_perfect package."""

import warnings

import numpy as np
import pandas as pd
import pytest

from rho_perfect.utils import validate_aggregated_df, validate_ratings_df
from rho_perfect import (
    calculate_rho_perfect,
    calculate_rho_perfect_from_ratings,
    split_raters_validation,
    split_ratings_validation,
)


# ---------------------------------------------------------------------------
# Synthetic ratings dataframes (fixtures).
# ---------------------------------------------------------------------------

def _make_ratings_df(
    n_items: int = 100,
    n_raters: int = 8,
    noise_std: float = 0.8,
    seed: int = 0,
) -> pd.DataFrame:
    """Synthetic per-rating DataFrame."""
    rng = np.random.default_rng(seed)
    true_quality = rng.uniform(1, 5, size=n_items)
    rows = []
    for i in range(n_items):
        for j in range(n_raters):
            rating = np.clip(true_quality[i] + rng.normal(0, noise_std), 1, 5)
            rows.append({
                "filename": f"item_{i:03d}",
                "rater_id": j,
                "rating": rating,
            })
    return pd.DataFrame(rows)


def _aggregate_ratings(df: pd.DataFrame) -> pd.DataFrame:
    """Helper to aggregate ratings DataFrame."""
    return (
        df.groupby("filename")["rating"]
        .agg(mean="mean", std=lambda x: x.std(ddof=1), n="count")
        .reset_index()
    )


@pytest.fixture
def ratings_df():
    return _make_ratings_df()


@pytest.fixture
def aggregated_df(ratings_df):
    return _aggregate_ratings(ratings_df)


# ---------------------------------------------------------------------------
# utils: validate_aggregated_df
# ---------------------------------------------------------------------------

class TestValidateAggregatedDf:
    def test_raises_on_missing_columns(self):
        df = pd.DataFrame({"filename": ["a"], "mean": [3.0]})
        with pytest.raises(ValueError, match="missing columns"):
            validate_aggregated_df(df)

    def test_warns_on_fewer_than_50_items(self):
        df = pd.DataFrame({
            "filename": [f"i{i}" for i in range(10)],
            "mean": [3.0] * 5 + [4.0] * 5,
            "std": [0.5] * 10,
            "n": [8] * 10,
        })
        with pytest.warns(UserWarning, match="at least 50"):
            validate_aggregated_df(df)

    def test_warns_on_fewer_than_3_ratings(self):
        df = pd.DataFrame({
            "filename": [f"i{i}" for i in range(60)],
            "mean": [3.0] * 30 + [4.0] * 30,
            "std": [0.5] * 60,
            "n": [8] * 59 + [2],
        })
        with pytest.warns(UserWarning, match="fewer than 3 ratings"):
            validate_aggregated_df(df)

    def test_no_warning_on_valid_input(self, aggregated_df):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            validate_aggregated_df(aggregated_df)


# ---------------------------------------------------------------------------
# utils: validate_ratings_df
# ---------------------------------------------------------------------------

class TestValidateRatingsDf:
    def test_raises_on_missing_columns(self):
        df = pd.DataFrame({"filename": ["a"]})
        with pytest.raises(ValueError, match="missing columns"):
            validate_ratings_df(df)

    def test_passes_on_valid_input(self, ratings_df):
        # Should not raise.
        validate_ratings_df(ratings_df)


# ---------------------------------------------------------------------------
# measure: calculate_rho_perfect
# ---------------------------------------------------------------------------

class TestCalculateRhoPerfect:
    def test_zero_noise_gives_one(self):
        """Identical ratings per item → no within-item variance → ceiling is 1."""
        df = pd.DataFrame({
            "filename": [f"i{i}" for i in range(60)],
            "mean": np.linspace(1, 5, 60),
            "std": np.zeros(60),
            "n": [8] * 60,
        })
        assert calculate_rho_perfect(df) == pytest.approx(1.0)

    def test_result_between_zero_and_one(self, aggregated_df):
        rp = calculate_rho_perfect(aggregated_df)
        assert 0.0 < rp <= 1.0

    def test_higher_noise_gives_lower_ceiling(self):
        low_noise = _aggregate_ratings(_make_ratings_df(noise_std=0.3))
        high_noise = _aggregate_ratings(_make_ratings_df(noise_std=1.5))
        assert calculate_rho_perfect(low_noise) > calculate_rho_perfect(high_noise)

    def test_raises_when_noise_dominates(self):
        """Constant mean, large std → Var(Ŷ) goes negative."""
        # The means can't be constant as the function will return 0; correlation
        # to a constant value doesn't make sense,
        df = pd.DataFrame({
            "filename": [f"i{i}" for i in range(60)],
            "mean": [3.0] * 30 + [4.0] * 30,
            "std": [2.0] * 60,
            "n": [4] * 60,
        })
        with pytest.raises(ValueError, match="non-positive"):
            calculate_rho_perfect(df)

    def test_more_raters_raises_ceiling(self):
        """More raters per item → smaller variance of the mean → higher ceiling."""
        few = _aggregate_ratings(_make_ratings_df(n_raters=4))
        many = _aggregate_ratings(_make_ratings_df(n_raters=20))
        assert calculate_rho_perfect(many) > calculate_rho_perfect(few)

    def test_ddof_parameter(self):
        """ddof=0 (population std) should give different result than ddof=1."""
        df = pd.DataFrame({
            "filename": [f"i{i}" for i in range(60)],
            "mean": np.linspace(1, 5, 60),
            "std": [0.5] * 60,
            "n": [8] * 60,
        })
        rp_sample = calculate_rho_perfect(df, ddof=1)
        rp_population = calculate_rho_perfect(df, ddof=0)
        # Population std will be corrected, and hence, higher than sample, so
        # ceiling should be higher for sample.
        assert rp_population < rp_sample

    def test_zero_variance_of_mean_raises_value_error(self):
        """If all items have the same mean rating, the ceiling should be zero."""
        df = pd.DataFrame({
            "filename": [f"i{i}" for i in range(60)],
            "mean": [3.0] * 60,
            "std": [0.5] * 60,
            "n": [8] * 60,
        })
        with pytest.raises(ValueError, match="All item means are identical."):
            calculate_rho_perfect(df)

    def test_only_one_rating_raises_value_error(self):
        """If all items have the same mean rating, the ceiling should be zero."""
        df = pd.DataFrame({
            "filename": [f"i{i}" for i in range(60)],
            "mean": [3.0] * 30 + [4.0] * 30,
            "std": [0.5] * 60,
            "n": [1] + [8] * 59,
        })
        with pytest.raises(ValueError, match="Some items have only one rating."):
            calculate_rho_perfect(df)

# ---------------------------------------------------------------------------
# measure: calculate_rho_perfect_from_ratings
# ---------------------------------------------------------------------------

class TestCalculateRhoPerfectFromRatings:
    def test_matches_aggregated_path(self, ratings_df, aggregated_df):
        """Both entry points should give the same result."""
        rp_ratings = calculate_rho_perfect_from_ratings(ratings_df)
        rp_aggregated = calculate_rho_perfect(aggregated_df)
        assert rp_ratings == pytest.approx(rp_aggregated)

    def test_result_between_zero_and_one(self, ratings_df):
        rp = calculate_rho_perfect_from_ratings(ratings_df)
        assert 0.0 < rp <= 1.0


# ---------------------------------------------------------------------------
# validation: split_raters_validation / split_ratings_validation
# ---------------------------------------------------------------------------

class TestValidation:
    def test_output_shape_split_raters(self, ratings_df):
        n_iterations = 5
        result = split_raters_validation(ratings_df, n_iterations=n_iterations)
        assert len(result) == n_iterations

    def test_output_shape_split_ratings(self, ratings_df):
        n_iterations = 5
        result = split_ratings_validation(ratings_df, n_iterations=n_iterations)
        assert len(result) == n_iterations

    def test_output_columns_split_raters(self, ratings_df):
        result = split_raters_validation(ratings_df, n_iterations=1)
        assert set(result.columns) == {
            "iteration", "rho_perfect", "rho_perfect_squared", "corr_y1_y2"
        }

    def test_output_columns_split_ratings(self, ratings_df):
        result = split_ratings_validation(ratings_df, n_iterations=1)
        assert set(result.columns) == {
            "iteration", "rho_perfect", "rho_perfect_squared", "corr_y1_y2"
        }

    def test_rho_perfect_squared_equals_square(self, ratings_df):
        """rho_perfect_squared column should be exactly rho_perfect²."""
        for fn in (split_raters_validation, split_ratings_validation):
            result = fn(ratings_df, n_iterations=3)
            np.testing.assert_array_almost_equal(
                result["rho_perfect_squared"].values,
                result["rho_perfect"].values ** 2,
            )

    def test_rho_perfect_squared_tracks_test_retest_split_raters(self, ratings_df):
        """Core statistical guarantee: ρ-Perfect² ≈ Corr(Y1, Y2)."""
        result = split_raters_validation(ratings_df, n_iterations=20)
        mean_predicted = result["rho_perfect_squared"].mean()
        mean_actual = result["corr_y1_y2"].mean()
        assert mean_predicted == pytest.approx(mean_actual, abs=0.03)

    def test_rho_perfect_squared_tracks_test_retest_split_ratings(self, ratings_df):
        """Core statistical guarantee: ρ-Perfect² ≈ Corr(Y1, Y2)."""
        result = split_ratings_validation(ratings_df, n_iterations=20)
        mean_predicted = result["rho_perfect_squared"].mean()
        mean_actual = result["corr_y1_y2"].mean()
        assert mean_predicted == pytest.approx(mean_actual, abs=0.03)

    def test_deterministic_with_same_seed(self, ratings_df):
        """Same seed → identical results."""
        for fn in (split_raters_validation, split_ratings_validation):
            r1 = fn(ratings_df, n_iterations=3, seed=123)
            r2 = fn(ratings_df, n_iterations=3, seed=123)
            pd.testing.assert_frame_equal(r1, r2)

    def test_different_seed_gives_different_results(self, ratings_df):
        """Different seeds → different splits → different numbers."""
        for fn in (split_raters_validation, split_ratings_validation):
            r1 = fn(ratings_df, n_iterations=3, seed=1)
            r2 = fn(ratings_df, n_iterations=3, seed=2)
            assert not r1["corr_y1_y2"].equals(r2["corr_y1_y2"])

    def test_split_raters_uses_all_raters(self, ratings_df):
        """Split-raters should partition all unique raters."""
        # With 8 raters, each split should have 4 raters in each group
        result = split_raters_validation(ratings_df, n_iterations=1, seed=42)
        # Can't directly test this without exposing internals, but we can check
        # that the function runs without error
        assert len(result) == 1

    def test_split_ratings_drops_odd_rating(self):
        """Items with odd ratings should have one dropped."""
        # Create a dataset where each item has 7 ratings (odd)
        low_ratings = [2.0 + np.random.randn() * 0.1 for _ in range(30 * 7)]
        high_ratings = [4.0 + np.random.randn() * 0.1 for _ in range(30 * 7)]
        df = pd.DataFrame({
            "filename": [f"item_{i}" for i in range(60) for _ in range(7)],
            "rater_id": [j for _ in range(60) for j in range(7)],
            "rating": low_ratings + high_ratings,
        })
        result = split_ratings_validation(df, n_iterations=1)
        # Should not raise and should produce valid output
        assert len(result) == 1
        assert 0 < result["rho_perfect"].iloc[0] <= 1