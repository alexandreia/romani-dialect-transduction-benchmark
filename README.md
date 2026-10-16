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

Research questions:

- **Q1:** How do different train/test splits affect measured performance in
  Romani dialect form transduction?
- **Q2:** Can simple baselines show when a split allows shortcut solutions?
- **Q3:** Under the stricter split, do ByT5-small and prompted LLM systems
  perform better than simple baselines?

## Current Research Framing

```text
Low-resource dialect-to-dialect form transduction for Romani varieties,
with special attention to inflectional paradigm slots.
```

The dataset and scripts are inspired by morphological reinflection/transduction
work, but adapted to dialect sample pairs. The project preserves comparative
dialect metadata such as `sample_id`, `variety`, `item_id`, and `meaning`.


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


## Tests

```bash
python -m pip install pytest
python -m pytest tests/ -v
```

Run the tests before every rebuild; do not rebuild on top of a failing test.

## Next Steps

1. **Phase 1 – dataset design:** finish the data-layer fixes (A1–A8), rebuild,
   and write the alignment coverage report (B2).
2. **Phase 2 – baselines:** fix the baseline issues (A9, B4–B6), rerun all
   baselines on all splits, before/after comparison (B3), confidence intervals.
3. **Phase 3 – ByT5-small** on the strict split.
4. **Phase 4 – prompted LLMs** (optional, scope decision D7).
5. **Phase 5 – analysis and presentation.**
