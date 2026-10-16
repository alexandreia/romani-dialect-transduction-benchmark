"""
Significance testing for dialect transduction results.

Key principle — the UNIT OF ANALYSIS: pair rows are not independent. The same
item appears in up to 42 dyads (and several slots), so testing on rows
(pseudo-replication) inflates the degrees of freedom and shrinks p-values.
All tests here first aggregate to a unit (default: item_id) and then test
across units.

Provided:
    paired_comparison   two systems scored on the SAME test rows
                        -> paired t-test (df = n_units - 1), paired bootstrap CI,
                           sign-flip permutation test
    welch_comparison    two INDEPENDENT groups (unequal variances, Welch df)
    holm_correction     adjust p-values when several comparisons are made
    summarize_runs      mean / SD over k folds or random seeds
    check_normality     Shapiro–Wilk, skewness, kurtosis of a sample
    check_equal_variance  Levene (median-centred, i.e. Brown–Forsythe)
    plot_paired_differences  histogram + Q–Q plot of the per-unit differences

Which assumption belongs to which test:
    paired t-test       per-unit DIFFERENCES roughly normal (equal variances NOT needed)
    Welch t-test        each group roughly normal or large; equal variances NOT needed
    Student t-test      normality AND equal variances (-> Levene) — prefer Welch
    bootstrap / permutation  no normality assumption; units must be independent

ALWAYS LOOK AT THE PLOT. Formal tests depend on sample size: with few units they
rarely detect real violations (low power); with many units they flag trivial
ones. The plot shows the shape; the test is a supporting number.

Reference for the bootstrap/permutation approach in NLP:
Dror et al. (2018), "The Hitchhiker's Guide to Testing Statistical Significance
in Natural Language Processing", ACL.

CLI example:
    python src/evaluation/significance.py \\
        --a reports/predictions/suffix_rule_with_backoff_heldout_item_split_test_exclude_check.csv \\
        --b reports/predictions/copy_source_heldout_item_split_test_exclude_check.csv \\
        --metric cer --unit item_id
"""

import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats

# Columns that identify one test row. A paired test is only valid if both
# systems were scored on exactly the same rows.
ROW_KEY_COLUMNS = ["pair_name", "item_id", "split_group", "slot"]


def row_keys(df):
    """Build one identity string per test row from the available key columns."""
    columns = [column for column in ROW_KEY_COLUMNS if column in df.columns]
    if "item_id" not in columns:
        raise ValueError("Predictions need at least an item_id column.")
    # astype(object).fillna("") first: with pandas 3 + pyarrow, astype(str) keeps
    # NaN as a missing value (e.g. empty `slot` for phrases), and "|".join fails.
    return df[columns].astype(object).fillna("").astype(str).agg("|".join, axis=1)


def check_same_rows(df_a, df_b):
    """Stop if the two prediction files do not cover exactly the same test rows."""
    keys_a, keys_b = row_keys(df_a), row_keys(df_b)
    if keys_a.duplicated().any() or keys_b.duplicated().any():
        raise ValueError("Duplicate row keys: cannot verify that both systems saw the same rows.")
    only_a, only_b = set(keys_a) - set(keys_b), set(keys_b) - set(keys_a)
    if only_a or only_b:
        raise ValueError(
            f"Systems were not scored on the same test rows "
            f"({len(only_a)} only in A, {len(only_b)} only in B): a paired test is invalid."
        )


def aggregate_by_unit(df, metric="cer", unit="item_id"):
    """One value per unit: the mean metric over all rows of that unit."""
    if unit == "row":
        return pd.Series(df[metric].to_numpy(), index=row_keys(df))
    return df.groupby(df[unit].astype(str))[metric].mean()


def paired_comparison(df_a, df_b, metric="cer", unit="item_id", n_boot=10_000, n_perm=10_000, seed=42):
    """
    Compare system A and system B on the same test rows.

    Returns a dict with the unit-level means, the mean difference (A - B), a
    paired t-test with df = n_units - 1, a paired bootstrap 95% CI of the
    difference, and a two-sided sign-flip permutation p-value.
    For CER (lower is better), a negative mean_diff means A is better.
    """
    check_same_rows(df_a, df_b)
    a, b = aggregate_by_unit(df_a, metric, unit).align(aggregate_by_unit(df_b, metric, unit), join="inner")
    diff = (a - b).to_numpy()
    n = len(diff)
    if n < 2:
        raise ValueError(f"Need at least 2 units, got {n}.")

    rng = np.random.default_rng(seed)

    if np.allclose(diff, 0):
        t_stat, p_t = 0.0, 1.0
    else:
        t_result = stats.ttest_rel(a.to_numpy(), b.to_numpy())
        t_stat, p_t = float(t_result.statistic), float(t_result.pvalue)

    # Paired bootstrap: resample UNITS with replacement, keep the pairing.
    boot_means = rng.choice(diff, size=(n_boot, n), replace=True).mean(axis=1)
    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])

    # Sign-flip permutation: under H0 the sign of each unit's difference is arbitrary.
    observed = abs(diff.mean())
    signs = rng.choice([-1.0, 1.0], size=(n_perm, n))
    permuted = np.abs((signs * diff).mean(axis=1))
    p_perm = (np.sum(permuted >= observed - 1e-12) + 1) / (n_perm + 1)

    normality = check_normality(diff)

    return {
        "metric": metric,
        "unit": unit,
        "n_units": n,
        "df": n - 1,
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
        "mean_diff": float(diff.mean()),
        "t": t_stat,
        "p_ttest": p_t,
        "bootstrap_ci_low": float(ci_low),
        "bootstrap_ci_high": float(ci_high),
        "p_permutation": float(p_perm),
        "diff_skewness": normality["skewness"],
        "diff_excess_kurtosis": normality["excess_kurtosis"],
        "diff_shapiro_p": normality["shapiro_p"],
    }


def check_normality(values):
    """Shapiro–Wilk test plus skewness and excess kurtosis (0 for a normal distribution)."""
    x = np.asarray(values, dtype=float)
    if len(x) < 3 or np.allclose(x, x[0]):
        # Shape statistics are undefined for (near-)constant data.
        nan = float("nan")
        return {"n": len(x), "skewness": nan, "excess_kurtosis": nan, "shapiro_w": nan, "shapiro_p": nan}
    result = {"n": len(x), "skewness": float(stats.skew(x)), "excess_kurtosis": float(stats.kurtosis(x))}
    if len(x) <= 5000:
        shapiro = stats.shapiro(x)
        result.update({"shapiro_w": float(shapiro.statistic), "shapiro_p": float(shapiro.pvalue)})
    else:
        result.update({"shapiro_w": float("nan"), "shapiro_p": float("nan")})
    return result


def check_equal_variance(*groups):
    """Levene's test centred on the median (Brown–Forsythe): robust to non-normal data."""
    result = stats.levene(*[np.asarray(g, dtype=float) for g in groups], center="median")
    return {
        "levene_w": float(result.statistic),
        "levene_p": float(result.pvalue),
        "group_variances": [float(np.var(g, ddof=1)) for g in groups],
    }


def plot_paired_differences(df_a, df_b, output_path, metric="cer", unit="item_id", labels=("A", "B")):
    """
    Save a figure with (1) a histogram of the per-unit differences and (2) a
    normal Q–Q plot. Points on the line = consistent with normality.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    a, b = aggregate_by_unit(df_a, metric, unit).align(aggregate_by_unit(df_b, metric, unit), join="inner")
    diff = (a - b).to_numpy()

    fig, (ax_hist, ax_qq) = plt.subplots(1, 2, figsize=(10, 4))
    ax_hist.hist(diff, bins=30, color="#4C72B0", edgecolor="white")
    ax_hist.axvline(0, color="black", linewidth=1)
    ax_hist.axvline(diff.mean(), color="#C44E52", linestyle="--", linewidth=1.5, label=f"mean = {diff.mean():.3f}")
    ax_hist.set_xlabel(f"{metric}({labels[0]}) − {metric}({labels[1]}) per {unit}")
    ax_hist.set_ylabel(f"number of {unit}s")
    ax_hist.set_title(f"Paired differences (n = {len(diff)})")
    ax_hist.legend()

    stats.probplot(diff, dist="norm", plot=ax_qq)
    ax_qq.set_title("Normal Q–Q plot of the differences")
    ax_qq.get_lines()[0].set_markersize(4)

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def welch_comparison(values_1, values_2):
    """
    Welch's t-test for two INDEPENDENT groups (unequal variances).

    Only use this when the groups share no units (e.g. no item appears in both).
    Returns t, p and the Welch–Satterthwaite degrees of freedom.
    """
    x, y = np.asarray(values_1, dtype=float), np.asarray(values_2, dtype=float)
    result = stats.ttest_ind(x, y, equal_var=False)
    var_x, var_y = x.var(ddof=1) / len(x), y.var(ddof=1) / len(y)
    df = (var_x + var_y) ** 2 / (var_x**2 / (len(x) - 1) + var_y**2 / (len(y) - 1))
    return {
        "n_1": len(x), "n_2": len(y),
        "mean_1": float(x.mean()), "mean_2": float(y.mean()),
        "t": float(result.statistic), "p": float(result.pvalue), "df_welch": float(df),
    }


def holm_correction(p_values):
    """Holm–Bonferroni adjusted p-values (same order as the input)."""
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adjusted = np.empty(m)
    running_max = 0.0
    for rank, index in enumerate(order):
        running_max = max(running_max, (m - rank) * p[index])
        adjusted[index] = min(1.0, running_max)
    return adjusted


def summarize_runs(values):
    """Mean, SD and n over folds or seeds (SD uses ddof=1)."""
    values = np.asarray(values, dtype=float)
    return {
        "n_runs": len(values),
        "mean": float(values.mean()),
        "sd": float(values.std(ddof=1)) if len(values) > 1 else float("nan"),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Item-level paired significance test between two systems.")
    parser.add_argument("--a", required=True, help="Scored predictions CSV of system A.")
    parser.add_argument("--b", required=True, help="Scored predictions CSV of system B (e.g. copy_source).")
    parser.add_argument("--metric", default="cer")
    parser.add_argument("--unit", default="item_id", help="item_id (default), pair_name, or row (not recommended).")
    parser.add_argument("--n-boot", type=int, default=10_000)
    parser.add_argument("--n-perm", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", help="Optional CSV to append the result row to.")
    parser.add_argument("--plot", help="Optional PNG path: histogram + Q-Q plot of the paired differences.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    df_a = pd.read_csv(args.a, keep_default_na=False)
    df_b = pd.read_csv(args.b, keep_default_na=False)
    result = paired_comparison(df_a, df_b, args.metric, args.unit, args.n_boot, args.n_perm, args.seed)
    result = {"system_a": os.path.basename(args.a), "system_b": os.path.basename(args.b), **result}

    for key, value in result.items():
        print(f"{key:>18}: {value:.4f}" if isinstance(value, float) else f"{key:>18}: {value}")

    if args.plot:
        plot_paired_differences(df_a, df_b, args.plot, args.metric, args.unit,
                                labels=(os.path.basename(args.a).split("_heldout")[0],
                                        os.path.basename(args.b).split("_heldout")[0]))
        print(f"Saved assumption plot to {args.plot}  <- look at it before trusting p_ttest")

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        row = pd.DataFrame([result])
        row.to_csv(args.output, mode="a", header=not os.path.exists(args.output), index=False)
        print(f"Appended result to {args.output}")
