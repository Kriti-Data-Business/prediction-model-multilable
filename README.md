# Multi-Label Complaint Classification Pipeline
### Telecom Complaint Tagging · 96 Tags · DistilBERT + CleanLab · Two-Approach Experimental Design

> **Project** — Master of Data Science (Project Stream), RMIT University  
> **Focus:** How data quality correction, smart sampling, and transformer-based architectures compare against classical multi-label baselines for real-world complaint classification at scale.

***

## Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Repository Structure](#repository-structure)
- [Approach 1 — DistilBERT Pipeline](#approach-1--distilbert-pipeline)
- [Approach 2 — CleanLab Ensemble Noise Correction](#approach-2--cleanlab-ensemble-noise-correction)
- [Smart Sampling Strategy](#smart-sampling-strategy)
- [Key Design Decisions](#key-design-decisions)
- [Results So Far](#results-so-far)
- [Setup & Installation](#setup--installation)
- [Evaluation Metrics](#evaluation-metrics)
- [Pipeline Architecture](#pipeline-architecture)

***

## Overview

This repository contains the full experimental pipeline for multi-label classification of telecom complaint data across **96 complaint tags**. Two parallel approaches are run and compared — one as a structured DistilBERT pipeline, and one as a noise-corrected ensemble pipeline using CleanLab — to document the architectural trade-offs and the impact of data quality correction.

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
3. **Annotation noise** — human-labelled complaint tags contain errors. In a 96-label multi-label setting, a single mislabelled complaint affects all 96 classifiers simultaneously through the shared training signal — structurally 96× worse than in binary classification.

***

## Repository Structure

```
prediction-model-multilable/
│
├── complaint_tagger_pipeline.ipynb           # Approach 1 — Full DistilBERT pipeline (15 blocks)
├── complaint_tagger_pipeline (1).ipynb       # Approach 1 — Iteration / variant run
├── complaint_tagger_pipeline (block0).ipynb  # Approach 1 — Block 0 smart sampling standalone
├── complaint_tagger_pipeline -output.ipynb   # Approach 1 — Run with outputs saved
│
├── complaint_tagger_DEFINITIVE.py            # Approach 1 — Production .py version (full pipeline)
├── complaint_tagger_DYNAMIC.py               # Approach 1 — Dynamic hyperparameter version
│
├── Approach2/
│   └── Clean_lab_lr-xg-nn-noisyperkeyword.py  # Approach 2 — CleanLab ensemble probe pipeline
│
├── cfpb_complaints_simulated.csv             # Simulated complaint dataset (development/testing)
├── tag_thresholds.npy                        # Per-tag prediction thresholds (saved from Block 11)
├── tag_info.csv                              # Tag metadata (tag names, frequencies, tier info)
│
├── Analysis.md                               # Experimental analysis notes
├── model_research.md                         # Literature and model research notes
├── onevsrest_method.md                       # OvR baseline methodology notes
├── Reference.md                              # Key references and citations
└── steps.md                                  # Pipeline step-by-step development notes
```

***

## Approach 1 — DistilBERT Pipeline

### Core Files
- [`complaint_tagger_pipeline.ipynb`](https://github.com/Kriti-Data-Business/prediction-model-multilable/blob/main/complaint_tagger_pipeline.ipynb) — Main pipeline notebook (15 blocks, end-to-end)
- [`complaint_tagger_DEFINITIVE.py`](https://github.com/Kriti-Data-Business/prediction-model-multilable/blob/main/complaint_tagger_DEFINITIVE.py) — Production `.py` version for Azure ML submission
- [`complaint_tagger_DYNAMIC.py`](https://github.com/Kriti-Data-Business/prediction-model-multilable/blob/main/complaint_tagger_DYNAMIC.py) — Variant with auto-diagnosed dynamic hyperparameters
- [`complaint_tagger_pipeline (block0).ipynb`](https://github.com/Kriti-Data-Business/prediction-model-multilable/blob/main/complaint_tagger_pipeline%20(block0).ipynb) — Block 0 smart sampling in isolation

### Architecture

A fine-tuned **DistilBERT** encoder (66M parameters) with a sigmoid classification head — one output per tag, trained simultaneously. The shared encoder naturally learns tag co-occurrence patterns across all 96 tags, which OvR architectures cannot.

```
Input text (Service + description + summary)
        │
   DistilBERT encoder (66M params)
        │
   [CLS] token → Dropout(0.3) → Linear(768 → 96) → Sigmoid
        │
   96 independent probability scores
        │
   Per-tag threshold (tag_thresholds.npy) → binary tag predictions
```

### Pipeline Blocks

| Block | Function |
|---|---|
| **Block 0** | Smart sampling — Phase 1: keyword quota, Phase 2: inverse-freq fill → 3,500 cases |
| **Block 2** | Load & parse — handles list / string / pipe-separated tag formats → `MultiLabelBinarizer` → y shape `(3500, 96)` |
| **Block 3** | Auto-diagnose — dynamically sets `GAMMA_NEG`, `KW_DROPOUT`, `EPOCHS`, `BATCH_SIZE` from data characteristics |
| **Block 4** | Drop rare tags (< `MIN_SAMPLES`) and now-empty rows |
| **Block 5** | CleanLab scoring (ensemble probe — see Approach 2) |
| **Block 6** | Train / val / test split — **split before augmentation** to prevent data leakage |
| **Block 7** | Three-layer imbalance handling: augment rare tags / undersample dominant-only / WeightedRandomSampler |
| **Block 8** | Dataset builder — keyword dropout (30–55%), adaptive label smoothing scaled to CleanLab noise score |
| **Block 9** | Model — DistilBERT + AsymmetricLoss + AdamW + linear warmup scheduler |
| **Block 10** | Training — per-epoch checkpoints, early stopping, best model auto-saved |
| **Block 11** | Per-tag threshold tuning on validation set → saved to `tag_thresholds.npy` |
| **Block 12** | Evaluation — Macro F1 (primary), Micro F1, Samples F1, per-tag classification report |
| **Block 13** | Human review export — uncertain / rare / high-confidence cases exported to CSV |
| **Block 14** | Feedback loop — merge human corrections → retrain from Block 2 |
| **Block 15** | Inference — `predict_tags(row_dict)` → returns tags + per-tag confidence scores |

### Key Training Choices

**AsymmetricLoss** (`γ⁻` auto-set from imbalance ratio detected in Block 3):
- Applies a high gradient penalty for confident wrong predictions on dominant tags
- Gentler penalty on rare tags — allows the model to learn from sparse signal
- Addresses imbalance at the loss level, not just at the data level

**Keyword Dropout (30–55% of training steps):**
- `keyword_tier1` and `keyword_tier2` are human-assigned categorical labels on each complaint
- Without dropout, the model shortcuts on these fields instead of reading the complaint `summary`
- Masking forces genuine semantic learning from complaint text

**Adaptive Label Smoothing:**
- Rows with a high CleanLab `noise_score` receive higher `ε` (label smoothing strength)
- Prevents the model from memorising uncertain annotations with full confidence

***

## Approach 2 — CleanLab Ensemble Noise Correction

### Core File
- [`Approach2/Clean_lab_lr-xg-nn-noisyperkeyword.py`](https://github.com/Kriti-Data-Business/prediction-model-multilable/blob/main/Approach2/Clean_lab_lr-xg-nn-noisyperkeyword.py) — Full CleanLab ensemble probe pipeline with per-keyword noise scoring

### Motivation

Human-annotated complaint tags contain labelling errors. In a 96-label multi-label setting, a single mislabelled complaint affects all 96 classifiers simultaneously through the shared DistilBERT encoder gradients — the damage is structurally **96× worse** than in binary classification. CleanLab quantifies this noise before any training begins.

### Why an Ensemble Probe (Not a Single Model)

Initial testing with a single OvR Logistic Regression as the CleanLab probe produced very low and unstable scores. The core issue: `multilabel_quality_scores()` requires reliable cross-validation probabilities to score label quality. A single LR on severely imbalanced 96-label data produces **collapsed probabilities for rare tags** — the model assigns near-zero probability to all rare tags, making CleanLab unable to distinguish genuine noise from imbalance-induced uncertainty.

**Solution:** An ensemble of three complementary classifiers probability-averaged before passing to CleanLab:

```python
models = [
    OneVsRestClassifier(LogisticRegression(max_iter=1000, class_weight='balanced')),
    OneVsRestClassifier(LinearSVC(class_weight='balanced')),
    OneVsRestClassifier(SGDClassifier(loss='modified_huber', class_weight='balanced'))
]
# Probability-average across all three → feed averaged probs to CleanLab
ensemble_probs = mean([model.predict_proba(X) for model in models], axis=0)
```

The ensemble stabilises the probability signal across the full 96-tag space, giving CleanLab reliable input to assign per-keyword noise scores.

### Per-Keyword Noise Scoring

A key design choice in Approach 2 (reflected in the filename `noisyperkeyword`) is that CleanLab noise scores are computed and analysed **per keyword group**, not just globally. This allows the pipeline to identify which keyword categories carry the most annotation noise — providing targeted signal for data cleaning and human review prioritisation.

### Row-Level vs Tag-Level Correction

Two correction granularities are implemented and compared as a thesis ablation:

**Row-level correction (primary):**
- `multilabel_quality_scores()` returns one noise score per complaint row
- Rows below threshold are removed entirely or down-weighted via adaptive label smoothing
- **Limitation:** If a complaint has 4 correct tags and 1 wrong tag, the entire row is discarded

**Tag-level correction (surgical extension):**
- `find_label_issues(multi_label=True)` identifies specific `(row_idx, tag_idx)` pairs likely mislabelled
- Only the suspicious tag is removed — the rest of the complaint is retained
- More data-efficient at 3,500 rows; documented as an ablation in the thesis

```python
from cleanlab.filter import find_label_issues

label_issues = find_label_issues(
    labels=y_train,            # shape (n_samples, 96)
    pred_probs=ensemble_probs, # shape (n_samples, 96)
    return_indices_ranked_by='self_confidence',
    multi_label=True
)
# Returns (row_idx, tag_idx) pairs → y_train[row_idx, tag_idx] = 0
```

### CleanLab Compute Cost

| Step | Time (CPU, 3,500 rows) |
|---|---|
| TF-IDF (15k features) | ~10 seconds |
| 5-fold cross-val (ensemble) | ~3–4 minutes |
| `multilabel_quality_scores()` | ~15 seconds |
| **Total** | **~5 minutes — one-time, cached** |

CleanLab costs ~5 minutes once and is cached. DistilBERT training costs 6–8 hours per run. The noise correction step is effectively free relative to training cost.

***

## Smart Sampling Strategy

The initial pipeline drew from 2 months of complaint data (3,500 cases). This produced keyword-level imbalance — rare keywords appeared fewer than 5 times in the training set, making those complaint types unlearnable regardless of model architecture.

**Revised strategy: 2 years of data → 3,500 cases via two phases:**

### Phase 1 — Keyword Quota
Guarantee a minimum of 10–20 samples per unique `keyword_group`. No keyword is excluded from training due to low frequency alone.

```python
quota_per_keyword = 15  # configurable
for keyword in unique_keywords:
    keyword_rows = df[df['keyword_group'] == keyword]
    n = min(len(keyword_rows), quota_per_keyword)
    shortlisted.append(keyword_rows.sample(n, random_state=42))
```

### Phase 2 — Inverse-Frequency Fill
Fill remaining quota to 3,500 using inverse-frequency weighted sampling. Dominant keywords are under-sampled; rare ones already at quota are excluded.

```python
remaining = 3500 - len(shortlisted)
weights = 1 / df_remaining['keyword_group'].map(keyword_freq)
fill = df_remaining.sample(remaining, weights=weights, random_state=42)
```

**Net effect:** Better distributional coverage across all 96 tags, more training signal for rare complaint types, and reduced keyword-level imbalance without artificially inflating rare tag counts.

***

## Key Design Decisions

### Why Split Before Augmentation?
The previous pipeline augmented training data first, then split. The validation set therefore contained augmented copies of training rows — inflating evaluated F1 by an estimated 0.08–0.12. The corrected pipeline always splits first (Block 6), then augments only the training portion (Block 7).

### Why OvR is a Baseline, Not the Final Model
OvR trains 96 completely independent binary classifiers. It cannot learn that `Account breach` and `Fraudulent transfer` co-occur, or that `Consumer protection` is almost always paired with other tags. Typical Macro F1 ceiling on 96-label imbalanced complaint text: **0.28–0.42**. This is a structural limit — no amount of hyperparameter tuning bridges the gap because the architecture fundamentally lacks shared representation across labels. OvR is correctly used here as a fast baseline and as the CleanLab probe model.

### Per-Tag Threshold Tuning
A single global threshold of 0.5 causes dominant tags to be over-predicted and rare tags to never fire (F1 = 0 on rare tags). Block 11 performs grid search per tag (0.10 → 0.90, step 0.05) on the validation set with hard constraints:
- Dominant tags: threshold forced ≥ 0.65 (prevents over-prediction)
- Rare tags: threshold forced ≤ 0.35 (improves recall)

Results saved to [`tag_thresholds.npy`](https://github.com/Kriti-Data-Business/prediction-model-multilable/blob/main/tag_thresholds.npy).

***

## Results So Far

| Run | Data | Model | Macro F1 |
|---|---|---|---|
| Initial run | 2-month sample, 3,500 cases | DistilBERT | **0.55** |
| Current run | 2-year sample, 3,500 cases (Phase 1+2) | DistilBERT | Running on Azure ML (~6–8 hrs) |
| Baseline | 2-year sample, 3,500 cases | OvR Logistic Regression | Running |
| Baseline | 2-year sample, 3,500 cases | XGBoost | Running |
| Approach 2 | 2-year sample + CleanLab ensemble correction | DistilBERT | In progress |

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

**To run Approach 1 locally:**
```bash
jupyter notebook "complaint_tagger_pipeline.ipynb"
```

**To run Approach 2 (CleanLab probe):**
```bash
python "Approach2/Clean_lab_lr-xg-nn-noisyperkeyword.py"
```

**To run on Azure ML:**  
Submit `complaint_tagger_DEFINITIVE.py` as a script job. Full pipeline runs in approximately 6–8 hours on CPU compute at 5 epochs.

***

## Evaluation Metrics

| Metric | What it measures | Why it matters here |
|---|---|---|
| **Macro F1** | Unweighted average F1 across all 96 tags | **Primary metric** — equal weight to rare and dominant tags |
| **Micro F1** | F1 weighted by tag frequency | Dominated by frequent tags — secondary only |
| **Samples F1** | Per-complaint average F1 | Measures per-row prediction quality |
| **Per-tag F1** | F1 for each individual tag | Identifies which specific tags need targeted improvement |

Macro F1 is the thesis primary metric because it gives rare tags equal weight to dominant ones — a model that predicts only dominant tags scores Macro F1 ≈ 0.00 on the rare tags, pulling the overall score down regardless of accuracy on common tags.

***

## Pipeline Architecture

```
RAW CSV (2 years of complaint data)
        │
        ▼
┌──────────────────────────────────┐
│  BLOCK 0 — Smart Sampling        │
│  Phase 1: keyword quota (10–20)  │
│  Phase 2: inverse-freq fill      │
│  Output: 3,500 balanced cases    │
└──────────────┬───────────────────┘
               │
               ▼
┌──────────────────────────────────┐
│  BLOCK 2 — Load & Parse          │
│  Auto-detect tag format          │
│  MultiLabelBinarizer → y(3500,96)│
└──────────────┬───────────────────┘
               │
               ▼
┌──────────────────────────────────┐
│  BLOCK 3 — Auto-Diagnose         │
│  Sets GAMMA_NEG, KW_DROPOUT      │
│  Sets EPOCHS, BATCH_SIZE, MAX_LEN│
└──────────────┬───────────────────┘
               │
               ▼
┌──────────────────────────────────┐
│  BLOCK 5 — CleanLab (Approach 2) │
│  3-model ensemble OvR probe      │
│  multilabel_quality_scores()     │
│  Per-keyword noise scoring       │
│  CACHED after first run          │
└──────────────┬───────────────────┘
               │
               ▼
┌──────────────────────────────────┐
│  BLOCK 6 — Split FIRST           │
│  80% train / 10% val / 10% test  │
│  Split before augmentation       │
└──────────────┬───────────────────┘
               │
               ▼
┌──────────────────────────────────┐
│  BLOCK 7 — 3-Layer Imbalance     │
│  Layer 1: Augment rare tags      │
│  Layer 2: Undersample dominant   │
│  Layer 3: WeightedRandomSampler  │
└──────────────┬───────────────────┘
               │
               ▼
┌──────────────────────────────────┐
│  BLOCK 9 — DistilBERT Model      │
│  [CLS] → Dropout → Linear(96)    │
│  AsymmetricLoss (γ⁻ auto-set)    │
│  AdamW + warmup scheduler        │
└──────────────┬───────────────────┘
               │
               ▼
┌──────────────────────────────────┐
│  BLOCK 11 — Per-Tag Thresholds   │
│  Grid search 0.10→0.90 per tag   │
│  Dominant ≥ 0.65, Rare ≤ 0.35   │
│  Saved → tag_thresholds.npy      │
└──────────────┬───────────────────┘
               │
               ▼
┌──────────────────────────────────┐
│  BLOCK 12 — Evaluation           │
│  Macro F1 (primary metric)       │
│  Per-tag classification report   │
└──────────────┬───────────────────┘
               │
               ▼
┌──────────────────────────────────┐
│  BLOCK 13 — Human Review Export  │
│  Uncertain / rare / high-conf    │
└──────────────┬───────────────────┘
               │  (corrections merged back)
               ▼
┌──────────────────────────────────┐
│  BLOCK 15 — Inference            │
│  predict_tags(row_dict)          │
│  Tags + confidence scores        │
└──────────────────────────────────┘
```

***

## Author

**Kriti Yadav**  
Master of Data Science (Project Stream) — RMIT University, Melbourne  
Supervised by Jonathan  
GitHub: [@Kriti-Data-Business](https://github.com/Kriti-Data-Business)
