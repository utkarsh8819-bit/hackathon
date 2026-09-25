# Amazon ML Challenge 2026: Business Entity Resolution System

An end-to-end, production-grade Entity Resolution (ER) system built for the Amazon ML Challenge "Business Entity Resolution Challenge".

## Architecture Overview

The system strictly follows a two-stage architecture:
1. **Stage A: Multi-Rule Inverted Index Blocking (High Recall)**
   - Normalizes raw business names (Unicode NFKD, legal suffix mapping, metaphone encoding).
   - Normalizes addresses (standardizes street abbreviations, extracts 6-digit Indian PIN / 5-digit US & French postal codes, house numbers, landmarks).
   - Inverted indexes over S2 + S3 candidate pool across 6 independent blocking rules (Exact name, informative token overlap, phonetic Metaphone, postal code, distinctive address tokens, character 3-grams).
   - Exports `output/candidate_pairs.tsv` as an audit artifact before any ML filtering.

2. **Stage B: Pairwise Matcher & Macro F_0.5 Threshold Optimizer (High Precision)**
   - Computes 25 pairwise similarity and structural features (Levenshtein, Jaro-Winkler, token-set, character n-gram cosine, postal agreement, house number agreement, open-set country equality, length discrepancies, multi-hot blocking rule triggers).
   - Trains LightGBM with tree-based interaction capture and class-balanced weighting.
   - Evaluates on a strictly grouped validation split (`source1_entity_id`) with zero data leakage.
   - Optimizes threshold to maximize the competition metric: **Per-Entity Macro-Averaged F_0.5** (including singleton credit).
   - Generates competition-compliant `output/matching_results.tsv`.

---

## Directory Structure

```
.
├── dataset/
│   ├── train/                  # train_source1.tsv, train_source2.tsv, train_source3.tsv, train_ground_truth.tsv
│   └── test/                   # test_source1.tsv, test_source2.tsv, test_source3.tsv
├── src/
│   ├── __init__.py
│   ├── config.py               # Paths, constants, schemas, hyperparameters
│   ├── io.py                   # TSV readers/writers and canonical schema representations
│   ├── validation.py           # Contract validation and ground truth parser
│   ├── normalization.py        # Text & name normalizer, legal suffix handler, phonetic encoder
│   ├── address_parser.py       # Address standardizer, postal code & landmark extractor
│   ├── blocking.py             # Multi-rule inverted index candidate generator
│   ├── pair_generation.py      # Positives, hard negatives, grouped train/val splitter
│   ├── features.py             # 25 pairwise similarity and structural features
│   ├── train.py                # Logistic Regression baseline & LightGBM training
│   ├── evaluate.py             # Official Per-Entity Macro F_0.5 evaluator & threshold sweep
│   ├── inference.py            # Test inference pipeline
│   ├── output.py               # 2-col and 4-col candidate & matching exporters
│   └── diagnostics.py          # Error bucketing and failure analysis
├── utils/
│   └── validate_submission.py  # Official competition validator
├── models/
│   ├── matcher.pkl             # Trained model artifact
│   └── threshold.json          # Optimal threshold and feature metadata
├── notebooks/
│   ├── 01_eda.ipynb            # Exploratory Data Analysis
│   ├── 02_blocking_analysis.ipynb # Blocking recall & reduction ratio analysis
│   └── 03_model_analysis.ipynb # Model evaluation, feature importance & error analysis
├── output/
│   ├── candidate_pairs.tsv     # Stage A candidates (competition format)
│   ├── candidate_pairs_internal.tsv # Internal 4-column debug table
│   └── matching_results.tsv    # Stage B predictions (leaderboard submission)
├── run_pipeline.py             # Unified end-to-end CLI orchestrator
├── Documentation_template.md   # Technical methodology report
├── requirements.txt            # Pinned dependencies
└── README.md
```

---

## Setup & Installation

Ensure Python 3.8+ (Python 3.10-3.13 tested) is installed:

```bash
pip install -r requirements.txt
```

---

## End-to-End Reproduction Instructions

### Option 1: One-Command Execution (Recommended)

Run the full end-to-end pipeline (training, threshold optimization, inference, validation, and submission packaging) with a single command:

```bash
python run_pipeline.py --train-dir dataset/train --test-dir dataset/test --team-name my_team --package
```

### Option 2: Step-by-Step Reproduction

#### 1. Train Matching Model & Optimize Threshold
```bash
python -m src.train --train-dir dataset/train --models-dir models --val-size 0.2
```
*Outputs: `models/matcher.pkl`, `models/threshold.json`.*

#### 2. Run Inference on Test Set
```bash
python -m src.inference --test-dir dataset/test --output-dir output --models-dir models
```
*Outputs: `output/candidate_pairs.tsv`, `output/matching_results.tsv`, `output/candidate_pairs_internal.tsv`.*

#### 3. Validate Submission Outputs
Run the validator script to verify all 6 hard competition rules:
```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test \
    --check-ids
```
*Must output `PASS — no blocking issues found. Safe to submit.`*

---

## Running Automated Test Suite

To verify all pipeline stages independently:

```bash
python -m pytest tests/ -v
```

Tests included:
- `tests/test_schema_and_io.py`: Schema validation, TSV delimiter preservation, singleton parsing.
- `tests/test_normalization.py`: Name & address normalization, legal suffixes, Indian/US/French postal codes.
- `tests/test_blocking.py`: Inverted index generation, candidate retrieval, blocking recall ceiling.
- `tests/test_evaluation.py`: Exact Per-Entity Macro F_0.5 formula and singleton edge cases.
- `tests/test_model_and_pipeline.py`: Full end-to-end pipeline execution and submission validator verification.
