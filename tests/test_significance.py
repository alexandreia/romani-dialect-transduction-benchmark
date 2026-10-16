"""Tests for src/evaluation/significance.py. Run: python -m pytest tests/"""
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src.evaluation.significance import (
    aggregate_by_unit, check_equal_variance, check_normality, holm_correction, paired_comparison,
    plot_paired_differences, row_keys, summarize_runs, welch_comparison,
)


def predictions(cer_by_item, rows_per_item=3):
    """Fake scored predictions: several rows (dyads) per item."""
    rows = []
    for item, cer in cer_by_item.items():
        for k in range(rows_per_item):
            rows.append({"pair_name": f"p{k}", "item_id": item, "split_group": f"{item}_form_1", "slot": "", "cer": cer})
    return pd.DataFrame(rows)


def test_aggregates_rows_to_items():
    df = predictions({"1": 0.2, "2": 0.4}, rows_per_item=5)
    assert len(df) == 10
    assert len(aggregate_by_unit(df, unit="item_id")) == 2     # df is n_items-1, not n_rows-1


def test_identical_systems_show_no_difference():
    df = predictions({str(i): i / 20 for i in range(20)})
    result = paired_comparison(df, df.copy())
    assert result["mean_diff"] == 0 and result["p_ttest"] == 1.0 and result["df"] == 19


def test_clear_improvement_is_detected():
    rng = np.random.default_rng(0)
    base = {str(i): v for i, v in enumerate(rng.uniform(0.3, 0.6, 40))}
    better = {k: v - 0.1 + rng.normal(0, 0.03) for k, v in base.items()}   # noisy, realistic gain
    result = paired_comparison(predictions(better), predictions(base))
    assert result["bootstrap_ci_high"] < 0                       # whole CI below zero: A better
    assert result["p_permutation"] < 0.01


def test_different_test_rows_are_rejected():
    with pytest.raises(ValueError, match="same test rows"):
        paired_comparison(predictions({"1": 0.1, "2": 0.2}), predictions({"1": 0.1, "3": 0.2}))


def test_welch_matches_scipy_and_returns_df():
    x, y = [0.1, 0.2, 0.3, 0.25], [0.5, 0.7, 0.4, 0.9, 0.6]
    result = welch_comparison(x, y)
    assert result["p"] == pytest.approx(stats.ttest_ind(x, y, equal_var=False).pvalue)
    assert 3 < result["df_welch"] < 7


def test_holm_correction_known_values():
    assert np.allclose(holm_correction([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])


def test_summarize_runs():
    assert summarize_runs([0.40, 0.42, 0.38]) == pytest.approx({"n_runs": 3, "mean": 0.40, "sd": 0.02})


def test_normality_check_flags_skewed_data():
    rng = np.random.default_rng(1)
    assert check_normality(rng.normal(size=300))["shapiro_p"] > 0.01
    assert check_normality(rng.exponential(size=300))["shapiro_p"] < 0.01


def test_equal_variance_check():
    rng = np.random.default_rng(2)
    same = check_equal_variance(rng.normal(0, 1, 200), rng.normal(0, 1, 200))
    different = check_equal_variance(rng.normal(0, 1, 200), rng.normal(0, 3, 200))
    assert same["levene_p"] > 0.01 and different["levene_p"] < 0.01


def test_plot_is_written(tmp_path):
    df_a, df_b = predictions({str(i): i / 30 for i in range(30)}), predictions({str(i): 0.5 for i in range(30)})
    path = plot_paired_differences(df_a, df_b, str(tmp_path / "plot.png"))
    assert (tmp_path / "plot.png").stat().st_size > 0


def test_missing_key_values_do_not_break_row_keys():
    # Phrase rows have no `slot`; pandas 3 + pyarrow keeps NaN after astype(str).
    df = pd.DataFrame({"item_id": [1, 2], "slot": [np.nan, "1SG"], "cer": [0.1, 0.2]})
    keys = row_keys(df)
    assert keys.tolist() == ["1|", "2|1SG"]
