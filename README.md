# Multi-Label Complaint Classification Pipeline
### Telecom Complaint Tagging · 96 Tags · DistilBERT + CleanLab · Two-Approach Experimental Design

> ** Project** — Master of Data Science (Project Stream), RMIT University  
> How data quality correction, smart sampling, and transformer-based architectures compare against classical multi-label baselines for real-world complaint classification at scale.

***

## Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Repository Structure](#repository-structure)
- [Approach 1 — DistilBERT Baseline Pipeline](#approach-1--distilbert-baseline-pipeline)
- [Approach 2 — CleanLab + Ensemble Noise Correction Pipeline](#approach-2--cleanlab--ensemble-noise-correction-pipeline)
- [Smart Sampling Strategy](#smart-sampling-strategy)
- [Key Design Decisions](#key-design-decisions)
- [Results So Far](#results-so-far)
- [Setup & Installation](#setup--installation)
- [Running on Azure ML](#running-on-azure-ml)
- [Evaluation Metrics](#evaluation-metrics)
- [Pipeline Architecture](#pipeline-architecture)

***

## Overview

This repository contains the full experimental pipeline for multi-label classification of telecom complaint data across **96 complaint tags**. The project runs two parallel experimental approaches:

| | Approach 1 | Approach 2 |
|---|---|---|
| **Architecture** | DistilBERT (fine-tuned) | Ensemble CleanLab probe → DistilBERT |
| **Data** | 3,500 cases (smart sampled, 2yr window) | Same dataset + noise correction |
| **Baselines** | OvR Logistic Regression, XGBoost | — |
| **Key technique** | AsymmetricLoss + per-tag threshold tuning | Row-level + tag-level CleanLab correction |
| **Current Macro F1** | 0.55 (initial 2-month run) | In progress |
| **Compute** | Azure ML (~6–8 hrs / 5 epochs) | Local + Azure ML |

***

## Problem Statement

Telecom complaint data presents three compounding challenges:

1. **Severe class imbalance** — dominant tags appear 180× more frequently than rare tags. A naive model learns to predict dominant tags exclusively, achieving near-zero F1 on rare but operationally important complaint types.
2. **Label co-occurrence** — complaints routinely carry multiple tags simultaneously (e.g., `Account breach` + `Fraudulent transfer`). Classical OvR classifiers treat each tag independently, missing the co-occurrence signal entirely.
3. **Annotation noise** — human-labelled complaint tags contain errors. In a 96-label multi-label setting, a single mislabelled complaint affects all 96 classifiers simultaneously through the shared training signal — the damage is structurally 96× worse than in binary classification.

***

## Repository Structure

```
prediction-model-multilable/
│
├── approach_1/
│   ├── complaint_tagger_pipeline.ipynb     # Full DistilBERT pipeline (15 blocks)
│   ├── baseline_ovr_lr.ipynb               # OvR Logistic Regression baseline
│   ├── baseline_xgboost.ipynb              # XGBoost baseline
│   └── smart_sampling.py                   # Phase 1 + Phase 2 sampling logic
│
├── approach_2/
│   ├── cleanlab_ensemble_probe.ipynb       # Ensemble OvR probe → CleanLab scoring
│   ├── cleanlab_tag_level_correction.ipynb # Tag-level surgical correction (extension)
│   └── distilbert_cleanlab_pipeline.ipynb  # Full pipeline post noise correction
│
├── data/
│   ├── shortlisted_3500.csv                # Smart-sampled working dataset
│   ├── train_split.csv                     # 80% train split (post augmentation)
│   ├── val_split.csv                       # 10% validation split
│   ├── test_split.csv                      # 10% held-out test split
│   └── tag_thresholds.json                 # Per-tag prediction thresholds
│
├── outputs/
│   ├── training_history.csv                # Loss + Macro F1 per epoch
│   ├── per_tag_report.csv                  # Per-tag precision/recall/F1
│   └── human_review/                       # Exported uncertain/rare/high-conf cases
│
├── requirements.txt
└── README.md
```

***

## Approach 1 — DistilBERT Baseline Pipeline

### Architecture

A fine-tuned **DistilBERT** encoder (66M parameters) with a sigmoid classification head — one output per tag, trained simultaneously. The shared encoder naturally learns tag co-occurrence patterns, which OvR architectures cannot.

```
Input text (Service + description + summary)
        │
   DistilBERT encoder
        │
   [CLS] token → Dropout(0.3) → Linear(768 → 96) → Sigmoid
        │
   96 independent probability scores
        │
   Per-tag threshold (tag_thresholds.json) → binary tag predictions
```

### Pipeline Blocks

| Block | Function |
|---|---|
| **Block 0** | Smart sampling (Phase 1: keyword quota, Phase 2: inverse-freq fill) |
| **Block 2** | Load & parse — handles list / string / pipe-separated tag formats |
| **Block 3** | Auto-diagnose — dynamically sets `GAMMA_NEG`, `KW_DROPOUT`, `EPOCHS`, `BATCH_SIZE` |
| **Block 4** | Drop rare tags (< `MIN_SAMPLES`) and now-empty rows |
| **Block 5** | CleanLab scoring (see Approach 2) |
| **Block 6** | Train / val / test split — **split before augmentation** to prevent data leakage |
| **Block 7** | Three-layer imbalance handling (augment / undersample / WeightedSampler) |
| **Block 8** | Dataset builder — keyword dropout (30–55%), adaptive label smoothing |
| **Block 9** | Model — DistilBERT + AsymmetricLoss + AdamW + warmup scheduler |
| **Block 10** | Training — per-epoch checkpoints, early stopping, best model saved |
| **Block 11** | Per-tag threshold tuning on validation set |
| **Block 12** | Evaluation — Macro F1, Micro F1, Samples F1, per-tag report |
| **Block 13** | Human review export (uncertain / rare / high-confidence cases) |
| **Block 14** | Feedback loop — merge corrections → retrain from Block 2 |
| **Block 15** | Inference — `predict_tags(row_dict)` → tags + confidence scores |

### Key Training Choices

**AsymmetricLoss** (`γ⁻` auto-set from imbalance ratio):
- Applies a high gradient penalty for confident wrong predictions on dominant tags
- Gentler penalty on rare tags — allows the model to learn from sparse signal without over-correction
- Directly addresses imbalance at the loss level, not just the data level

**Keyword Dropout (30–55%):**
- `keyword_tier1` and `keyword_tier2` fields are human-assigned categorical labels
- Without dropout, the model shortcuts on these fields instead of reading the complaint text
- Masking forces the model to learn from the `summary` and `description` fields

**Adaptive Label Smoothing:**
- Rows with high CleanLab noise scores receive higher `ε` (label smoothing strength)
- Prevents the model from memorising uncertain annotations with full confidence

***

## Approach 2 — CleanLab + Ensemble Noise Correction Pipeline

### Motivation

Human-annotated complaint tags contain labelling errors. A single mislabelled complaint in a 96-label setting affects all classifiers through the shared DistilBERT encoder gradients — the damage is structurally 96× worse than in binary classification. CleanLab quantifies this noise before training begins.

### Why an Ensemble Probe (Not a Single Model)

Initial testing with a single OvR Logistic Regression as the CleanLab probe model produced very low and unstable F1 scores. The core issue: `multilabel_quality_scores()` requires reliable cross-validation probabilities to score label quality accurately. A single LR on severely imbalanced 96-label data produces **collapsed probabilities for rare tags** — the model assigns near-zero probability to all rare tags, making CleanLab unable to distinguish genuine noise from imbalance.

**Solution:** An ensemble of three complementary classifiers:

```python
models = [
    OneVsRestClassifier(LogisticRegression(max_iter=1000, class_weight='balanced')),
    OneVsRestClassifier(LinearSVC(class_weight='balanced')),
    OneVsRestClassifier(SGDClassifier(loss='modified_huber', class_weight='balanced'))
]
# Probability-average across models → feed to CleanLab
```

The ensemble stabilises the probability signal across the full tag space, giving CleanLab reliable input to score each row's label quality.

### Row-Level vs Tag-Level Correction

Two correction granularities are implemented and compared:

**Row-level correction (primary):**
- `multilabel_quality_scores()` → one noise score per complaint row
- Rows below threshold are **removed entirely** or down-weighted via adaptive label smoothing
- Simple, fast, conservative
- **Limitation:** If a complaint has 4 correct tags and 1 wrong tag, the entire row is discarded — including the 4 correct labels

**Tag-level correction (extension — surgical):**
- `find_label_issues(multi_label=True)` → identifies specific `(row_idx, tag_idx)` pairs that are likely mislabelled
- Only the suspicious tag is removed — the rest of the complaint is retained
- More data-efficient, particularly important at 3,500 rows
- Documented as a direct ablation in the thesis to measure whether surgical correction yields measurable F1 improvement over whole-row removal

```python
from cleanlab.filter import find_label_issues

label_issues = find_label_issues(
    labels=y_train,            # shape (n_samples, 96)
    pred_probs=ensemble_probs, # shape (n_samples, 96)
    return_indices_ranked_by='self_confidence',
    multi_label=True
)
# Returns (row_idx, tag_idx) pairs
# y_train[row_idx, tag_idx] = 0  → surgical tag removal
```

***

## Smart Sampling Strategy

The initial pipeline used 2 months of complaint data (3,500 cases, stratified). This produced keyword-level imbalance — rare keywords appeared fewer than 5 times in the training set, making those complaint types unlearnable.

**Revised strategy: 2 years of data → 3,500 cases via two phases:**

### Phase 1 — Keyword Quota
Guarantee a minimum of 10–20 samples per unique `keyword_group`. This ensures no keyword is excluded from training due to frequency bias alone.

```python
quota_per_keyword = 15  # configurable
for keyword in unique_keywords:
    keyword_rows = df[df['keyword_group'] == keyword]
    n = min(len(keyword_rows), quota_per_keyword)
    shortlisted.append(keyword_rows.sample(n, random_state=42))
```

### Phase 2 — Inverse-Frequency Fill
Fill the remaining quota to reach 3,500 total using inverse-frequency weighted sampling. Dominant keywords are under-sampled; rare ones already at quota are excluded from this phase.

```python
remaining = 3500 - len(shortlisted)
weights = 1 / df_remaining['keyword_group'].map(keyword_freq)
fill = df_remaining.sample(remaining, weights=weights, random_state=42)
```

**Result:** Better distributional coverage across all 96 tags, more training signal for rare complaint types, and reduced keyword-level imbalance without artificially inflating rare tag counts.

***

## Key Design Decisions

### Why Split Before Augmentation?
The previous pipeline augmented training data first, then split. This meant the validation set contained augmented copies of training rows — inflating evaluated F1 by an estimated 0.08–0.12. The corrected pipeline always splits first, then augments only the training portion.

### Why OvR is a Baseline, Not the Final Model
OvR trains 96 completely independent binary classifiers. It cannot learn that `Account breach` and `Fraudulent transfer` co-occur, or that `Consumer protection` is almost always paired with other tags. Typical Macro F1 ceiling on 96-label imbalanced complaint text: **0.28–0.42**. This ceiling is structural — no amount of hyperparameter tuning bridges the gap because the architecture fundamentally lacks shared representation across labels.

### Per-Tag Threshold Tuning
A single threshold of 0.5 causes dominant tags to be over-predicted (high recall, low precision) and rare tags to never fire (F1 = 0). Grid search per tag (0.10 → 0.90, step 0.05) on the validation set, with hard constraints:
- Dominant tags: threshold forced ≥ 0.65
- Rare tags: threshold forced ≤ 0.35

***

## Results So Far

| Run | Data | Model | Macro F1 |
|---|---|---|---|
| Initial | 2-month sample (3,500) | DistilBERT | **0.55** |
| Current | 2-year sample (3,500, Phase 1+2) | DistilBERT | Running on Azure ML |
| Baseline | 2-year sample (3,500) | OvR LR | Running |
| Baseline | 2-year sample (3,500) | XGBoost | Running |
| Approach 2 | 2-year sample + CleanLab | DistilBERT | In progress |

***

## Setup & Installation

```bash
git clone https://github.com/Kriti-Data-Business/prediction-model-multilable.git
cd prediction-model-multilable
pip install -r requirements.txt
```

**Core dependencies:**
```
torch>=2.0.0
transformers>=4.38.0
cleanlab>=2.6.0
scikit-learn>=1.4.0
pandas>=2.0.0
numpy>=1.26.0
xgboost>=2.0.0
imbalanced-learn>=0.12.0
```

***

## Running on Azure ML

The full DistilBERT pipeline (Approach 1, 5 epochs) currently runs in approximately **6–8 hours** on Azure ML CPU compute. Checkpoints are saved per epoch to `outputs/checkpoints/epoch_N.pt` — the run can be resumed from any epoch without restarting.

```bash
# Submit to Azure ML
az ml job create --file azureml_job.yml --workspace-name <your-workspace>
```

To run locally (GPU recommended):
```bash
jupyter notebook approach_1/complaint_tagger_pipeline.ipynb
```

***

## Evaluation Metrics

| Metric | What it measures | Why it matters here |
|---|---|---|
| **Macro F1** | Unweighted average F1 across all 96 tags | Primary metric — gives equal weight to rare and dominant tags |
| **Micro F1** | F1 weighted by tag frequency | Dominated by frequent tags — secondary metric only |
| **Samples F1** | Per-complaint average F1 | Measures per-row prediction quality |
| **Per-tag F1** | F1 for each individual tag | Identifies which specific tags need targeted improvement |

***

## Pipeline Architecture

```
RAW CSV (2 years, ~20k rows)
        │
        ▼
┌──────────────────────────────┐
│  BLOCK 0 — Smart Sampling    │
│  Phase 1: keyword quota      │
│  Phase 2: inverse-freq fill  │
│  Output: 3,500 cases         │
└─────────────┬────────────────┘
              │
              ▼
┌──────────────────────────────┐
│  BLOCK 2 — Load & Parse      │
│  MultiLabelBinarizer → y     │
│  Shape: (3500, 96)           │
└─────────────┬────────────────┘
              │
              ▼
┌──────────────────────────────┐
│  BLOCK 3 — Auto-Diagnose     │
│  Sets GAMMA_NEG, KW_DROPOUT  │
│  Sets EPOCHS, BATCH_SIZE     │
└─────────────┬────────────────┘
              │
              ▼
┌──────────────────────────────┐
│  BLOCK 5 — CleanLab (Appr 2) │
│  Ensemble OvR probe (3 mdls) │
│  multilabel_quality_scores() │
│  → per-row noise_score       │
│  CACHED after first run      │
└─────────────┬────────────────┘
              │
              ▼
┌──────────────────────────────┐
│  BLOCK 6 — Split             │
│  80% train / 10% val / 10%   │
│  Split BEFORE augmentation   │
└─────────────┬────────────────┘
              │
              ▼
┌──────────────────────────────┐
│  BLOCK 7 — Imbalance (3 Lyr) │
│  Augment rare tags           │
│  Undersample dominant-only   │
│  WeightedRandomSampler       │
└─────────────┬────────────────┘
              │
              ▼
┌──────────────────────────────┐
│  BLOCK 9 — DistilBERT        │
│  [CLS] pooling → Linear(96)  │
│  AsymmetricLoss (γ⁻ auto)    │
│  AdamW + warmup scheduler    │
└─────────────┬────────────────┘
              │
              ▼
┌──────────────────────────────┐
│  BLOCK 11 — Thresholds       │
│  Per-tag grid search         │
│  Dominant ≥ 0.65             │
│  Rare ≤ 0.35                 │
└─────────────┬────────────────┘
              │
              ▼
┌──────────────────────────────┐
│  BLOCK 12 — Evaluation       │
│  Macro F1 (primary)          │
│  Per-tag classification rpt  │
└─────────────┬────────────────┘
              │
              ▼
┌──────────────────────────────┐
│  BLOCK 13 — Human Review     │
│  Export uncertain cases      │
│  Export rare-tag cases       │
└─────────────┬────────────────┘
              │ (corrections merged back)
              ▼
┌──────────────────────────────┐
│  BLOCK 15 — Inference        │
│  predict_tags(row_dict)      │
│  Returns tags + confidence   │
└──────────────────────────────┘
```

***

## Author

**Kriti Yadav**  
Master of Data Science (Project Stream) — RMIT University, Melbourne  
Supervised by Jonathan  
GitHub: [@Kriti-Data-Business](https://github.com/Kriti-Data-Business)
