# Evaluation

This folder contains scoring, error analysis, and reporting scripts for the
Romani dialect form transduction benchmark.

The main evaluation principle is:

```text
Do not rely on one aggregate score.
```

Aggregate scores are useful as a reference, but interpretation should be
stratified by:

```text
form_type
slot
sample pair
alignment_key_type
quality_flag
```

## Primary Metrics

Primary metrics:

```text
CER
normalized Levenshtein similarity
chrF
```

These are more suitable for dialectal form comparison than exact match alone
because they capture partial character-level similarity.

Secondary metrics:

```text
exact match
copy rate
degeneration rate
suffix similarity
stem similarity
```

## Scripts

### `metrics.py`

Scores prediction files and can also run a quick copy-source demo through the
standardized dataset loader.

Run a demo copy-source evaluation:

```bash
.venv/bin/python src/evaluation/metrics.py \
  --demo-copy-baseline \
  --split data/splits/heldout_item_split.json \
  --partition test \
  --exclude-check
```

Score an existing prediction CSV:

```bash
.venv/bin/python src/evaluation/metrics.py \
  --predictions reports/predictions/copy_source_heldout_item_test_exclude_check.csv \
  --summary-output reports/model_comparison/copy_source_grouped_summary.csv
```

Required prediction columns:

```text
model_input_form
model_target_form
prediction
```

### `error_analysis.py`

Creates diagnostic tables from a scored prediction CSV.

Example:

```bash
.venv/bin/python src/evaluation/error_analysis.py \
  --predictions reports/predictions/copy_source_heldout_item_test_exclude_check.csv \
  --output-folder reports/error_analysis/copy_source_heldout_item_test_exclude_check
```

Outputs:

```text
worst_predictions.csv
error_by_form_type.csv
error_by_slot.csv
error_by_pair.csv
error_by_alignment_key_type.csv
error_by_quality_flag.csv
```

### `report.py`

Builds a benchmark report from existing baseline summaries and error-analysis
CSV files.

It does not run baselines or models.

Preview what it would read:

```bash
.venv/bin/python src/evaluation/report.py --dry-run
```

Generate the report:

```bash
.venv/bin/python src/evaluation/report.py
```

If the report outputs already exist and you intentionally want to replace them:

```bash
.venv/bin/python src/evaluation/report.py --overwrite
```

Outputs:

```text
reports/benchmark_summary.md
reports/model_comparison_table.csv
reports/error_analysis_by_category.csv
```

## Recommended Workflow

1. Run baselines with `src/baselines/run_all.py`.
2. Save row-level predictions with `--save-predictions`.
3. Run `src/evaluation/error_analysis.py` on important prediction files.
4. Run `src/evaluation/report.py --dry-run`.
5. Run `src/evaluation/report.py` to write the report.

## Reporting Policy

Report aggregate scores as reference only.

Primary claims should come from stratified comparisons such as:

```text
CER by form_type
chrF by slot
copy rate by pair
error rates by alignment_key_type
```

This matters because lexemes, phrase/sentence forms, and inflected paradigm
forms behave differently in this benchmark.

