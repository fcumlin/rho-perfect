"""ρ-Perfect: Correlation ceiling for subjectively rated datasets.

Based on:
    Cumlin, F., "ρ-Perfect: Correlation Ceiling for Subjective Evaluation
    Datasets", ICASSP 2026.
"""

from rho_perfect.measure import (
    calculate_rho_perfect,
    calculate_rho_perfect_from_ratings,
)
from rho_perfect.validation import (
    split_raters_validation,
    split_ratings_validation,
)

__version__ = "0.1.0"

__all__ = [
    "calculate_rho_perfect",
    "calculate_rho_perfect_from_ratings",
    "split_raters_validation",
    "split_ratings_validation",
]
