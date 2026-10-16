# Baselines

This folder contains non-neural baselines for the Romani dialect form
transduction benchmark.

The purpose of these baselines is to establish a scientific floor before neural
models are introduced. A neural model is only meaningful if it can beat simple
copying, memorization, nearest-neighbor analogy, and lightweight suffix rules.

## Scripts

### `simple.py`

Runs one baseline at a time.

Available baselines:

```text
copy_source
target_form_lookup
nearest_source_form
suffix_rule
suffix_rule_with_backoff
character_rewrite_rule
```

Run one baseline without writing files:

```bash
.venv/bin/python src/baselines/simple.py \
  --baseline copy_source \
  --split data/splits/heldout_item_split.json \
  --test-partition test \
  --exclude-check
```

Save row-level predictions and a grouped summary:

```bash
.venv/bin/python src/baselines/simple.py \
  --baseline copy_source \
  --split data/splits/heldout_item_split.json \
  --test-partition test \
  --exclude-check \
  --prediction-output reports/predictions/copy_source_heldout_item_test_exclude_check.csv \
  --summary-output reports/model_comparison/copy_source_heldout_item_test_exclude_check_grouped_summary.csv
```

The script refuses to overwrite existing output files unless `--overwrite` is
passed.

### `run_all.py`

Runs all current baselines and writes one aggregate comparison table.

Run the main heldout-item benchmark:

```bash
.venv/bin/python src/baselines/run_all.py \
  --split data/splits/heldout_item_split.json \
  --test-partition test \
  --exclude-check
```

Also save row-level predictions:

```bash
.venv/bin/python src/baselines/run_all.py \
  --split data/splits/heldout_item_split.json \
  --test-partition test \
  --exclude-check \
  --save-predictions
```

Run the heldout-pair benchmark:

```bash
.venv/bin/python src/baselines/run_all.py \
  --split data/splits/heldout_pair_split.json \
  --test-partition test \
  --exclude-check \
  --save-predictions
```

Run the stricter target-form benchmark:

```bash
.venv/bin/python src/baselines/run_all.py \
  --split data/splits/heldout_target_item_slot_split.json \
  --test-partition test \
  --exclude-check \
  --save-predictions
```

The script refuses to overwrite existing output files unless `--overwrite` is
passed.

## Baseline Roles

`copy_source`

Copies the source form directly as the prediction. This measures how much of the
task is already solved by identical or near-identical forms across dialect
samples.

`target_form_lookup`

Memorizes target forms by target sample, `item_id`, and `slot` from the training
data. This is a leakage/memorization diagnostic, not a fair form-only model. On
a proper heldout-item split, it should collapse toward copying because test item
IDs are unseen.

`nearest_source_form`

Finds the closest training source form by edit distance and reuses its target
form. This is an analogical baseline: it asks whether similar-looking source
forms in training suggest useful target forms.

`suffix_rule`

Learns simple source-suffix to target-suffix correspondences, conditioned on
target sample and slot. This tests whether lightweight morphology-like rewrite
rules help beyond copying.

`suffix_rule_with_backoff`

Learns suffix rewrites with richer context and backoff. It first tries rules
specific to the source-target sample pair, form type, and slot, then backs off
to variety-level, target-sample, slot-level, and global rules. This is a
stronger morphology-like baseline than `suffix_rule`, but it still only rewrites
the final token and does not use the target form at prediction time.

`character_rewrite_rule`

Learns short character-level source substring -> target substring rewrite rules.
This is a diagnostic non-neural baseline for internal spelling/sound
correspondences. It is more flexible than `suffix_rule`, but still much simpler
than ByT5 because it applies frequent local rewrite rules without learning full
sequence context.

## Outputs

Aggregate baseline summaries are written to:

```text
reports/model_comparison/
```

Row-level scored predictions are written to:

```text
reports/predictions/
```

Prediction files can be passed to:

```text
src/evaluation/error_analysis.py
```

and summaries can be collected by:

```text
src/evaluation/report.py
```

## Recommended Baseline Order

1. Run all baselines on `heldout_item_split`.
2. Run all baselines on `heldout_pair_split`.
3. Run all baselines on `heldout_target_item_slot_split`.
4. Inspect whether `character_rewrite_rule` helps or hurts relative to copying.
5. Save row-level predictions for each important run.
6. Run error analysis on the prediction files.
7. Generate the benchmark report.
