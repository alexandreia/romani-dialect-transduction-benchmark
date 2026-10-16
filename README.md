# Romani Dialect Form Transduction Benchmark

This project is building a research benchmark for low-resource Romani dialect
form transduction.

The current focus is not neural modeling yet. The current focus is a clean,
auditable dataset layer and simple baselines that make the task scientifically
well-defined.

In practical terms, the project asks:

```text
Given a form in one Romani dialect sample, can we predict the corresponding
form in another dialect sample?
```

Example:

```text
source sample: RO-001 Romungro
target sample: RO-008 Kaldaraš
source form:   me kerav
target form:   me kǝrav
slot:          1SG
```

## Project Documents

- Proposal: [proposal/Project_Proposal_Andreia_Alexa.pdf](proposal/Project_Proposal_Andreia_Alexa.pdf)
  — *Evaluating Romani dialect sample-to-sample form transduction: Benchmark
  design, baselines and data leakage*
- Review checklist (immediate fixes, broader fixes, open decisions):
  [docs/review_checklist.md](docs/review_checklist.md)
- Parked ideas: [docs/ideas/](docs/ideas/)

Research questions (from the proposal):

- **Q1:** How do different train/test splits affect measured performance in
  Romani dialect form transduction?
- **Q2:** Can simple baselines show when a split allows shortcut solutions?
- **Q3:** Under the stricter split, do ByT5-small and prompted LLM systems
  perform better than simple baselines?

## Current Research Framing

The strongest current framing is:

```text
Low-resource dialect-to-dialect form transduction for Romani varieties,
with special attention to inflectional paradigm slots.
```

The dataset and scripts are inspired by morphological reinflection/transduction
work, but adapted to dialect sample pairs. The project preserves comparative
dialect metadata such as `sample_id`, `variety`, `item_id`, and `meaning`.

## Repository Structure

```text
src/
  data/          normalize.py, build_canonical.py, build_pairs.py,
                 splitters.py, load_pairs.py, README.md
  baselines/     simple.py, run_all.py, README.md
  evaluation/    metrics.py, error_analysis.py, report.py, README.md
  models/        train_byt5.py, README.md

tests/
  test_normalize.py        pronoun/slot inference and normalization tests

scripts/
  archive_outputs.sh       snapshot processed data + splits before a rebuild
  slurm/                   ByT5 cluster job files

data/
  *_items.csv              raw samples (RO-001 ... RO-072)
  processed/
    canonical_forms.csv
    pairs/
    archive/
  splits/
    archive/

proposal/                  project proposal (PDF)
docs/                      command guide, findings, framing, review checklist, ideas/
reports/                   error_analysis/, model_comparison/, predictions/
archive/                   legacy_2026-10-07/, additional_data/
```

Older exploratory scripts, notebooks, old model outputs, and previous generated
alignment/similarity outputs have been moved to:

[archive/legacy_2026-10-07](archive/legacy_2026-10-07)

The current benchmark pipeline lives under `src/`.

## Setup

Use the project virtual environment:

```bash
source .venv/bin/activate
```

If recreating the environment from scratch, install at least:

```bash
python -m pip install -r requirements.txt
```

For the optional ByT5 neural pilot, install:

```bash
python -m pip install -r requirements-neural.txt
```

## Dataset Pipeline

Detailed dataset documentation is in:

[src/data/README.md](src/data/README.md)

A command-by-command guide is in:

[docs/command_guide.md](docs/command_guide.md)

Current result interpretation is in:

[docs/current_findings.md](docs/current_findings.md)

Historical results are logged in:

[docs/historical_results.md](docs/historical_results.md)

The current research framing is in:

[docs/research_framing.md](docs/research_framing.md)

Neural model documentation is in:

[src/models/README.md](src/models/README.md)

The short version:

```text
raw CSV files
  -> canonical form table
  -> pairwise transduction files
  -> benchmark split definitions
  -> standardized dataset loader
```

### 1. Build Canonical Forms

```bash
.venv/bin/python src/data/build_canonical.py
```

Creates:

```text
data/processed/canonical_forms.csv
```

Canonical rows preserve:

```text
form_raw
form_normalized
sample_id
variety
item_id
meaning
form_index
form_type
person
number
slot
quality_flag
```

Important design decisions:

- Unicode normalization is conservative.
- Diacritics are preserved, with one exception for matching/modeling: the two
  schwa code points `ǝ` (U+01DD) and `ə` (U+0259) are unified to `ə` (U+0259)
  in `form_normalized`. `form_raw` is never changed. (Checklist A6, 2026-10-13.)
- Pronoun detection ignores parentheses inside the first token, so `(v)on`
  is read as `von` (3PL); Ursari `on` (3PL) and `ou` (3SG) are in the pronoun
  map. (Checklist A1, 2026-10-13.)
- Backslash-separated cells are split into separate forms.
- `slot` is inferred only when possible. Note: it is currently inferred for
  every form, including non-paradigm phrases (to be restricted to paradigm
  cells, checklist A2).
- `CHECK` rows are retained for later manual review.
- `tense` is not included because it is not reliably annotated.

### 2. Build Pairwise Transduction Files

```bash
.venv/bin/python src/data/build_pairs.py
```

Creates ordered sample-to-sample files in:

```text
data/processed/pairs/
```

Pair files preserve both raw and normalized forms:

```text
source_form_raw
source_form_normalized
target_form_raw
target_form_normalized
```

Alignment policy:

```text
if slot is known:
    align by item_id + slot
else:
    align by item_id + form_index
```

The `alignment_key_type` column records whether a row used:

```text
slot
form_index_fallback
```

Both merges are inner joins: a form that finds no partner under its key is
dropped from the pair file. These dropped rows are not yet counted or logged
(checklist A3). The `quality_flag` of a pair row currently comes from the
source form only (checklist A4).

### 3. Build Benchmark Splits

```bash
.venv/bin/python src/data/splitters.py
```

Creates:

```text
data/splits/random_item_split.json
data/splits/heldout_item_split.json
data/splits/heldout_pair_split.json
data/splits/heldout_target_item_slot_split.json
```

Split meanings:

- `random_item_split`: sanity check by `item_id`.
- `heldout_item_split`: main within-pair generalization split, stratified by
  dominant `form_type`.
- `heldout_pair_split`: cross-sample/dialect-pair generalization split.
- `heldout_target_item_slot_split`: stricter target-form split that prevents
  the same `target_sample_id + item_id + slot` group from appearing in more than
  one partition.

For `heldout_pair_split`, inverse directions are grouped together. For example,
if `RO-001 -> RO-008` is held out, then `RO-008 -> RO-001` is also held out.

Build only the stricter split:

```bash
.venv/bin/python src/data/splitters.py \
  --splits heldout_target_item_slot_split
```

### 4. Load Datasets Consistently

Use:

```bash
.venv/bin/python src/data/load_pairs.py \
  --split data/splits/heldout_item_split.json \
  --partition test \
  --exclude-check \
  --preview 5
```

Future baselines and models should use:

```python
from src.data.load_pairs import load_pair_dataset
```

The loader always creates model-facing columns:

```text
model_input_form
model_target_form
```

These are mapped from:

```text
source_form_normalized
target_form_normalized
```

This prevents future scripts from accidentally using raw forms or handling
splits inconsistently.

## Evaluation Metrics

Detailed evaluation documentation is in:

[src/evaluation/README.md](src/evaluation/README.md)

The core metrics module is:

[src/evaluation/metrics.py](src/evaluation/metrics.py)

Primary metrics:

```text
CER
normalized Levenshtein similarity
chrF
```

Secondary metrics:

```text
exact match
copy rate
degeneration rate
suffix similarity
stem similarity
```

Run a copy-source demo through the metrics module:

```bash
.venv/bin/python src/evaluation/metrics.py \
  --demo-copy-baseline \
  --split data/splits/heldout_item_split.json \
  --partition test \
  --exclude-check
```

## Error Analysis

The error-analysis module is:

[src/evaluation/error_analysis.py](src/evaluation/error_analysis.py)

It consumes a scored prediction CSV and writes diagnostic tables:

```text
worst_predictions.csv
error_by_form_type.csv
error_by_slot.csv
error_by_pair.csv
error_by_alignment_key_type.csv
error_by_quality_flag.csv
```

Example:

```bash
.venv/bin/python src/evaluation/error_analysis.py \
  --predictions reports/predictions/copy_source_heldout_item_test_exclude_check.csv \
  --output-folder reports/error_analysis/copy_source_heldout_item_test_exclude_check
```

Current copy-source error analysis lives in:

```text
reports/error_analysis/copy_source_heldout_item_test_exclude_check/
```

One current finding is that copy-source performs much better on split paradigm
forms than on lexemes or longer phrase/sentence items:

```text
form_type            CER
paradigm_form        0.2535
phrase_or_sentence   0.5363
lexeme               0.9402
```

## Baselines

Detailed baseline documentation is in:

[src/baselines/README.md](src/baselines/README.md)

The simple baseline runner is:

[src/baselines/simple.py](src/baselines/simple.py)

Implemented baselines:

```text
copy_source
target_form_lookup
nearest_source_form
suffix_rule
suffix_rule_with_backoff
character_rewrite_rule
```

Baseline roles:

- `copy_source`: measures how much of the task is solved by identical or
  near-identical forms across dialects.
- `target_form_lookup`: memorization/leakage diagnostic using target sample,
  `item_id`, and `slot`.
- `nearest_source_form`: analogical reuse baseline.
- `suffix_rule`: simple inflectional ending correspondence baseline.
- `suffix_rule_with_backoff`: stronger suffix baseline with slot, form-type,
  sample, and variety backoff.
- `character_rewrite_rule`: short character-level rewrite baseline for internal
  spelling/sound correspondences.

Run one baseline:

```bash
.venv/bin/python src/baselines/simple.py \
  --baseline copy_source \
  --split data/splits/heldout_item_split.json \
  --test-partition test \
  --exclude-check
```

Run all current baselines:

```bash
.venv/bin/python src/baselines/run_all.py \
  --split data/splits/heldout_item_split.json \
  --test-partition test \
  --exclude-check
```

This writes:

```text
reports/model_comparison/heldout_item_split_test_exclude_check_baseline_summary.csv
```

To also save row-level scored predictions:

```bash
.venv/bin/python src/baselines/run_all.py \
  --split data/splits/heldout_item_split.json \
  --test-partition test \
  --exclude-check \
  --save-predictions
```

Useful baseline commands to run yourself:

```bash
# Main heldout-item benchmark
.venv/bin/python src/baselines/run_all.py \
  --split data/splits/heldout_item_split.json \
  --test-partition test \
  --exclude-check \
  --save-predictions

# Heldout-pair benchmark
.venv/bin/python src/baselines/run_all.py \
  --split data/splits/heldout_pair_split.json \
  --test-partition test \
  --exclude-check \
  --save-predictions

# Stricter target-form benchmark
.venv/bin/python src/baselines/run_all.py \
  --split data/splits/heldout_target_item_slot_split.json \
  --test-partition test \
  --exclude-check \
  --save-predictions
```

The scripts refuse to overwrite existing report files unless `--overwrite` is
passed.

## Benchmark Report

The report wrapper is:

[src/evaluation/report.py](src/evaluation/report.py)

It reads existing CSV outputs from:

```text
reports/model_comparison/
reports/error_analysis/
```

and writes:

```text
reports/benchmark_summary.md
reports/model_comparison_table.csv
reports/error_analysis_by_category.csv
```

It does not run baselines or models. It only summarizes files that already
exist.

Preview what it would read:

```bash
.venv/bin/python src/evaluation/report.py --dry-run
```

Generate the report:

```bash
.venv/bin/python src/evaluation/report.py
```

If you intentionally want to replace an existing report:

```bash
.venv/bin/python src/evaluation/report.py --overwrite
```

## Baseline Snapshot (historical, pre-fix)

These numbers were produced on 2026-10-07, **before** the A1/A6 normalization
fixes. They are kept for the before/after comparison and must be regenerated
after the rebuild (checklist A14, B3).

Setting: `heldout_item_split`, test partition, CHECK rows excluded.

```text
baseline              CER     NormLev  chrF    Exact
copy_source           0.4894  0.6420   0.4933  0.0992
target_form_lookup    0.4894  0.6420   0.4933  0.0992
nearest_source_form   0.5419  0.5191   0.3717  0.0034
suffix_rule           0.4960  0.6350   0.4885  0.0990
```

The files in `reports/` still use the old baseline names:
`meaning_lookup` = `target_form_lookup`, and
`nearest_neighbor_edit_distance` = `nearest_source_form`.

On the heldout-item split, `target_form_lookup` collapses to `copy_source`
because test item IDs are unseen. That is expected and confirms that the split
blocks simple item memorization.

## CHECK Rows

`CHECK` rows are not automatically bad data. They are rows where a split form
could not be assigned a confident person/number/slot.

Many are non-verbal split sets, such as:

```text
number lists
possessive paradigms
oblique pronoun forms
sentence alternatives
lexical alternatives
```

Current policy:

- keep `CHECK` rows in canonical and pair files;
- allow later scripts to filter them with `--exclude-check`;
- manually review them later rather than forcing uncertain labels.

## Archive Policy

Generated outputs should not be silently overwritten.

When schemas or split logic change, old generated outputs are archived under:

```text
data/processed/archive/
data/splits/archive/
```

Many scripts also refuse to overwrite existing outputs unless `--overwrite` is
passed.

Before a rebuild, snapshot the current outputs with a label saying *why*:

```bash
bash scripts/archive_outputs.sh before_A2_A3
# -> data/processed/archive/<YYYY-MM-DD>_before_A2_A3/
#    data/splits/archive/<YYYY-MM-DD>_before_A2_A3/
```

Then rebuild in order:

```bash
python src/data/build_canonical.py --overwrite
python src/data/build_pairs.py --overwrite
python src/data/splitters.py --overwrite
```

## Tests

```bash
python -m pip install pytest
python -m pytest tests/ -v
```

Run the tests before every rebuild; do not rebuild on top of a failing test.

## Next Steps

Work follows the proposal's phases; item-level tasks live in
[docs/review_checklist.md](docs/review_checklist.md).

1. **Phase 1 – dataset design:** finish the data-layer fixes (A1–A8), rebuild,
   and write the alignment coverage report (B2).
2. **Phase 2 – baselines:** fix the baseline issues (A9, B4–B6), rerun all
   baselines on all splits, before/after comparison (B3), confidence intervals.
3. **Phase 3 – ByT5-small** on the strict split.
4. **Phase 4 – prompted LLMs** (optional, scope decision D7).
5. **Phase 5 – analysis and presentation.**
