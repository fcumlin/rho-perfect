# $\rho$-Perfect: Correlation Ceiling for Subjective Evaluation Datasets

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

Estimate the maximum achievable correlation between model predictions and human ratings, given the inherent noise in subjective data.

**Reference:** Cumlin, F., "Rho-Perfect: Correlation Ceiling for Subjective Evaluation Datasets", ICASSP 2026.

## Installation

```bash
# From GitHub
pip install git+https://github.com/fcumlin/rho-perfect.git

# With conda (install dependencies first)
conda create -n myenv python=3.10 numpy pandas scipy
conda activate myenv
pip install git+https://github.com/fcumlin/rho-perfect.git
```

## Quick Start

```python
import pandas as pd
from rho_perfect import calculate_rho_perfect

# The data: one row per item with aggregated statistics.
ratings = pd.DataFrame({
    'filename': ['item_001', 'item_002', ...],
    'mean': [3.2, 4.1, ...],      # mean rating per item
    'std': [0.5, 0.3, ...],       # sample std per item
    'n': [8, 8, ...]              # number of ratings per item
})

rho_perfect = calculate_rho_perfect(ratings)
print(f"$\rho$-Perfect = {rho_perfect:.3f}")

# Compare to a model on the same data.
model_pcc = 0.85  # pcc = Pearsons correlation coefficient.
if model_pcc >= 0.95 * rho_perfect:
    print("Model is close to ceiling. Improve data quality for further gains.")
else:
    print(f"Model can improve. Gap to ceiling: {rho_perfect - model_pcc:.3f}")
```

**From individual ratings:**
```python
from rho_perfect import calculate_rho_perfect_from_ratings

# Raw ratings: one row per rating
ratings = pd.DataFrame({
    'filename': ['item_001', 'item_001', 'item_002', ...],
    'rater_id': [0, 1, 0, ...],
    'rating': [3.0, 3.5, 4.0, ...]
})

rho_perfect = calculate_rho_perfect_from_ratings(ratings)
```

## Assumptions

$\rho$-Perfect estimates a correlation ceiling under the following assumptions:

- Ratings are **conditionally independent given an item**
- Rating noise may vary across items (**heteroscedasticity**)
- Each item has **at least 3 ratings** to allow estimation of within-item variance (altough in practice, it might be ok if some have fewer ratings per item as we take the mean of the variance over all items)
- The dataset exhibits non-zero between-item variability (i.e., the mean varies over items)

Violations of these assumptions may lead to unreliable estimates. The
implementation emits warnings when common failure modes are detected.

## Definition

**Definition 2.1 ($\rho$-Perfect):** Given a subjectively rated dataset $\mathcal{D} = \\{x_i, r_i^{(j)}\\}$, where $x_i$ is the $i$'th item and $r_i^{(j)}$ is the $(j)$'th rating on item $i$, the $\rho$-Perfect metric is given by

$$\rho\text{-Perfect} \triangleq \sqrt{\frac{\text{Var}(\hat{Y})}{\text{Var}(Y)}}$$

where $\text{Var}(\hat{Y})$ is the variance of a perfect predictor, and $\text{Var}(Y)$ is the variance of the average ratings per item. They are estimated by:

$$\text{Var}(Y) = \frac{1}{n-1} \sum_{i=1}^n (y_i - \bar{y})^2$$

$$\text{Var}(\hat{Y}) = \text{Var}(Y) - \frac{1}{n} \sum_{i=1}^n \frac{1}{m_i(m_i-1)} \sum_{j=1}^{m_i} (r_i^{(j)} - y_i)^2$$

where $y_i = \frac{1}{m_i}\sum_{j=1}^{m_i} r_i^{(j)}$ is the average rating for item $i$, and $m_i$ is the number of ratings for item $i$.

**Interpretation:** $\rho$-Perfect estimates the **maximum achievable Pearson correlation** between
any model and the mean human ratings on a given subjectively rated dataset.

## API

### `calculate_rho_perfect(subjective_statistics, ddof=1)`
Calculate $\rho$-Perfect from aggregated statistics.
- **Input:** DataFrame with columns `filename`, `mean`, `std`, `n`
- **Output:** float (0 < $\rho$ ≤ 1)
- **Warnings:** < 50 items or < 3 ratings per item

### `calculate_rho_perfect_from_ratings(subjective_ratings)`
Calculate $\ho$-Perfect from individual ratings.
- **Input:** DataFrame with columns `filename`, `rater_id`, `rating`
- **Output:** float (0 < $\rho$ ≤ 1)

### Validation Functions
```python
from rho_perfect import split_raters_validation, split_ratings_validation

# Validate $\rho$-Perfect^2 ≈ test-retest correlation (Section 3.1 of paper)
results = split_raters_validation(df, n_iterations=10, seed=42)
results = split_ratings_validation(df, n_iterations=10, seed=42)
```

## Testing

```bash
pip install -e ".[dev]"
pytest tests/
```

## Citation

```bibtex
@inproceedings{cumlin2026rhoperfect,
  title={Rho-Perfect: Correlation Ceiling for Subjective Evaluation Datasets},
  author={Cumlin, Fredrik},
  booktitle={ICASSP 2026},
  year={2026}
}
```

## License

MIT License - see [LICENSE](LICENSE) file for details.
