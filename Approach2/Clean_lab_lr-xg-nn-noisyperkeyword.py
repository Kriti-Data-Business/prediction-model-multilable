
# =============================================================================
# BLOCK 5 — CleanLab with 3 Probe Models (CORRECT: models run WITHIN CleanLab)
#
# Architecture:
#   Each tag is routed to a CleanLearning instance wrapping the right model.
#   CleanLearning internally handles:
#     - cross-validation
#     - out-of-fold probability generation
#     - label issue detection per tag
#
#   Tag routing:
#     Dominant tags (≥ 100 examples) → CleanLearning( OvR Logistic Regression )
#     Mid-freq tags (30–99 examples) → CleanLearning( MLP Neural Net )
#     Rare tags     (< 30 examples)  → CleanLearning( XGBoost )
#
#   Final step:
#     Stitch per-tag pred_probs → multilabel_label_quality_scores()
#     Drop noisy per-complaint-tag labels surgically → y_cleaned
#
# Inputs  (from Block 2 in memory):
#     df        : dataframe with Service, keyword_group, summary, tier_2
#     y         : (n_samples, 96) binary numpy array
#     mlb       : fitted MultiLabelBinarizer
#
# Outputs:
#     y_cleaned              : (n_samples, 96) cleaned binary label matrix
#     df_cleaned             : dataframe with tier_2_cleaned column
#     shortlisted_3500_cleaned.csv
#     y_cleaned.npy
# =============================================================================

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model    import LogisticRegression
from sklearn.neural_network  import MLPClassifier
from sklearn.calibration     import CalibratedClassifierCV
from xgboost                 import XGBClassifier

from cleanlab.classification import CleanLearning   # ← the correct CleanLab entry point
from cleanlab.multilabel_classification.label_issues import (
    multilabel_label_quality_scores
)

# =============================================================================
# SECTION 0 — CONFIGURATION
# =============================================================================

DOMINANT_THRESH = 100    # ≥ 100 examples → OvR Logistic Regression
RARE_THRESH     = 30     # < 30 examples  → XGBoost
                         # 30–99           → MLP

NOISE_THRESHOLD = 0.3    # label quality score below this → noisy label
                         # lower = stricter cleaning (try 0.25 if over-cleaning)

CV_FOLDS        = 5      # CleanLearning uses this internally for cross-val
TFIDF_FEATURES  = 15000

CLEANED_CSV = "shortlisted_3500_cleaned.csv"
CLEANED_NPY = "y_cleaned.npy"

# =============================================================================
# SECTION 1 — TAG ROUTING
# Assign each of the 96 tags to exactly one model group based on frequency
# =============================================================================

tag_names  = list(mlb.classes_)
n_samples  = len(df)
n_tags     = len(tag_names)
tag_counts = y.sum(axis=0).astype(int)

dominant_idx = [i for i, c in enumerate(tag_counts) if c >= DOMINANT_THRESH]
rare_idx     = [i for i, c in enumerate(tag_counts) if c < RARE_THRESH]
mid_idx      = [i for i, c in enumerate(tag_counts)
                if RARE_THRESH <= c < DOMINANT_THRESH]

print(f"Dataset        : {n_samples} complaints × {n_tags} tags")
print(f"Dominant (LR)  : {len(dominant_idx)} tags")
print(f"Mid-freq (MLP) : {len(mid_idx)} tags")
print(f"Rare (XGB)     : {len(rare_idx)} tags")

assert len(dominant_idx) + len(mid_idx) + len(rare_idx) == n_tags, \
    "Tag routing mismatch — some tags unrouted"

# =============================================================================
# SECTION 2 — BUILD SHARED TF-IDF FEATURES
# All 3 CleanLearning instances share the same feature matrix X_tfidf
# =============================================================================

print("\nBuilding TF-IDF features...")

df["_probe_text"] = (
    df["Service"].fillna("").str.strip() + " | " +
    df["keyword_group"].fillna("").str.strip() + " | " +
    df["summary"].fillna("").str.strip()
)

vectorizer = TfidfVectorizer(
    max_features=TFIDF_FEATURES,
    ngram_range=(1, 2),
    sublinear_tf=True,
    min_df=2
)
X_tfidf = vectorizer.fit_transform(df["_probe_text"])
print(f"TF-IDF matrix: {X_tfidf.shape}")

# =============================================================================
# SECTION 3 — DEFINE THE 3 BASE MODELS
# These are standard sklearn-compatible estimators.
# CleanLearning wraps each one — it calls .fit() and .predict_proba()
# internally during its cross-validation loop.
#
# NOTE: CleanLearning requires predict_proba() to exist on the base model.
# XGBoost and MLP have it natively; LogisticRegression has it natively.
# CalibratedClassifierCV adds it to any estimator that lacks it.
# =============================================================================

def make_lr():
    """OvR Logistic Regression — for dominant tags (≥100 examples)"""
    return LogisticRegression(
        C=1.0,
        class_weight="balanced",  # handles within-tag binary imbalance
        max_iter=1000,
        solver="saga",            # fastest for large sparse TF-IDF input
        random_state=42
    )

def make_xgb():
    """XGBoost with isotonic calibration — for rare tags (<30 examples)
    Calibration is critical here: raw XGBoost probabilities on rare tags
    are overconfident, which misleads CleanLab's noise scoring."""
    return CalibratedClassifierCV(
        XGBClassifier(
            n_estimators=100,
            max_depth=4,          # shallow to avoid overfitting on small-n
            learning_rate=0.1,
            scale_pos_weight=15,  # compensates for heavy negative:positive ratio
            use_label_encoder=False,
            eval_metric="logloss",
            verbosity=0,
            random_state=42
        ),
        cv=3,
        method="isotonic"         # isotonic regression better than sigmoid for trees
    )

def make_mlp():
    """MLP with sigmoid calibration — for mid-frequency tags (30–99 examples)
    CleanLearning calls predict_proba() which is available via
    CalibratedClassifierCV wrapping the MLP."""
    return CalibratedClassifierCV(
        MLPClassifier(
            hidden_layer_sizes=(256, 128),
            activation="relu",
            max_iter=200,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42
        ),
        cv=3,
        method="sigmoid"          # Platt scaling for neural nets
    )

# =============================================================================
# SECTION 4 — RUN CLEANLEARNING PER TAG (3 MODEL GROUPS)
#
# CleanLearning.fit(X, y_binary) does the following INTERNALLY:
#   1. Runs CV_FOLDS cross-validation using the wrapped model
#   2. Generates out-of-fold predicted probabilities (unbiased)
#   3. Uses Confident Learning algorithm to identify likely label errors
#   4. Returns label_issues_ attribute with per-sample issue flags
#
# We run one CleanLearning per tag (binary classification per tag).
# This is the CORRECT way to use CleanLab — models run inside CleanLab,
# not outside it.
#
# pred_probs matrix is assembled FROM CleanLearning's internal probabilities
# via .predict_proba() on the full dataset after fitting.
# =============================================================================

pred_probs   = np.zeros((n_samples, n_tags), dtype=np.float32)
cl_instances = {}   # store fitted CleanLearning objects for inspection later

def run_cleanlearning_group(tag_indices, model_factory, group_name):
    """
    Fits one CleanLearning instance per tag in the group.
    Extracts out-of-fold predicted probabilities from each instance.
    Stores fitted CleanLearning objects in cl_instances dict.

    Parameters
    ----------
    tag_indices   : list of int — column indices in y for this group
    model_factory : callable — returns a fresh base estimator
    group_name    : str — for progress display
    """
    for step, tag_idx in enumerate(tag_indices):
        tag_name   = tag_names[tag_idx]
        y_binary   = y[:, tag_idx]          # shape (n_samples,) — binary per tag

        # Skip tags with no positive examples (can't learn anything)
        if y_binary.sum() == 0:
            print(f"  [{group_name}] SKIP {tag_name} — no positive examples")
            continue

        if step % 10 == 0:
            print(f"  [{group_name}] Tag {step+1}/{len(tag_indices)}: {tag_name} "
                  f"(n_pos={int(y_binary.sum())})")

        # ── CleanLearning wraps the model and runs CV internally ──
        cl = CleanLearning(
            clf=model_factory(),         # your model goes IN here
            cv_n_folds=CV_FOLDS,         # CleanLearning does the cross-val
            verbose=False
        )

        # .fit() triggers internal CV, OOF probability generation,
        # and Confident Learning label issue detection — all inside CleanLab
        cl.fit(X_tfidf, y_binary)

        # Extract the OOF predicted probabilities CleanLearning used internally
        # These are the same unbiased probabilities CleanLab used to find issues
        # shape: (n_samples, 2) → we take column 1 (P(label=1))
        oof_probs = cl.predict_proba(X_tfidf)[:, 1]
        pred_probs[:, tag_idx] = oof_probs.astype(np.float32)

        # Store for later inspection (e.g. which tags had most issues)
        cl_instances[tag_idx] = cl


print("\n=== Running CleanLearning: OvR Logistic Regression (Dominant Tags) ===")
run_cleanlearning_group(dominant_idx, make_lr, "OvR-LR")

print("\n=== Running CleanLearning: XGBoost (Rare Tags) ===")
run_cleanlearning_group(rare_idx, make_xgb, "XGBoost")

print("\n=== Running CleanLearning: MLP (Mid-Frequency Tags) ===")
run_cleanlearning_group(mid_idx, make_mlp, "MLP")

# Clip to valid probability range
pred_probs = np.clip(pred_probs, 1e-6, 1 - 1e-6)

print(f"\npred_probs assembled: {pred_probs.shape}")

# =============================================================================
# SECTION 5 — MULTILABEL QUALITY SCORING
# Now that pred_probs is assembled from CleanLearning's internal probabilities,
# pass to multilabel_label_quality_scores() for per-tag per-complaint scoring.
#
# This gives a (n_samples, n_tags) matrix where:
#   Score → 1.0 : model and label agree  → likely correct
#   Score → 0.0 : model and label disagree → likely noisy label
# =============================================================================

print("\n=== Computing multilabel label quality scores ===")

label_quality_scores = multilabel_label_quality_scores(
    labels=y,
    pred_probs=pred_probs
)

print(f"Quality scores shape : {label_quality_scores.shape}")
print(f"Score range          : {label_quality_scores.min():.4f} – {label_quality_scores.max():.4f}")
print(f"Mean quality score   : {label_quality_scores.mean():.4f}")
print(f"% scores below {NOISE_THRESHOLD} : "
      f"{(label_quality_scores < NOISE_THRESHOLD).mean()*100:.1f}%")

# =============================================================================
# SECTION 6 — SURGICAL LABEL CLEANING (per complaint-tag, not per row)
#
# Only flip label 1→0 where:
#   (a) quality score is below NOISE_THRESHOLD  (CleanLab flags as noisy)
#   (b) the current label IS 1                  (there's something to remove)
#
# We never add labels (flip 0→1) — only remove suspected wrong ones.
# Whole rows are only dropped if ALL their tags were removed (very rare).
# =============================================================================

print("\n=== Surgical label cleaning ===")

y_cleaned  = y.copy().astype(np.int8)
noise_mask = (label_quality_scores < NOISE_THRESHOLD) & (y == 1)
y_cleaned[noise_mask] = 0

# ── Summary ──
total_before = int(y.sum())
total_after  = int(y_cleaned.sum())
dropped      = total_before - total_after

print(f"Label assignments before : {total_before}")
print(f"Label assignments after  : {total_after}")
print(f"Labels dropped           : {dropped}  ({dropped/total_before*100:.1f}%)")
print(f"Complaints corrected     : {int(noise_mask.any(axis=1).sum())}")

# Per-tag noise breakdown
tags_dropped  = noise_mask.sum(axis=0)
tag_noise_df  = pd.DataFrame({
    "tag"           : tag_names,
    "model_group"   : [
        "LR"  if i in dominant_idx else
        "XGB" if i in rare_idx     else "MLP"
        for i in range(n_tags)
    ],
    "original_count": tag_counts,
    "labels_dropped": tags_dropped.astype(int),
    "pct_dropped"   : (tags_dropped /
                       (tag_counts.astype(float) + 1e-9) * 100).round(1)
}).sort_values("pct_dropped", ascending=False)

print("\nTop 15 noisiest tags:")
print(tag_noise_df.head(15).to_string(index=False))

# =============================================================================
# SECTION 7 — REMOVE EMPTY ROWS
# Complaints left with zero tags after cleaning are uninformative — drop them.
# =============================================================================

empty_rows = (y_cleaned.sum(axis=1) == 0)
print(f"\nRows with zero tags after cleaning: {int(empty_rows.sum())} → removed")

df_cleaned = df[~empty_rows].copy().reset_index(drop=True)
y_cleaned  = y_cleaned[~empty_rows]

print(f"Final cleaned dataset: {len(df_cleaned)} rows × {y_cleaned.shape[1]} tags")

# =============================================================================
# SECTION 8 — RECONSTRUCT tier_2_cleaned COLUMN
# Convert binary matrix back to list of tag name strings.
# Original tier_2 is preserved for comparison and audit.
# =============================================================================

def binary_to_tags(row, classes):
    return [classes[i] for i, v in enumerate(row) if v == 1]

df_cleaned["tier_2_cleaned"] = [
    binary_to_tags(y_cleaned[i], tag_names)
    for i in range(len(df_cleaned))
]

print("\nSample before vs after:")
for idx in range(min(8, len(df_cleaned))):
    before = df_cleaned["tier_2"].iloc[idx]
    after  = df_cleaned["tier_2_cleaned"].iloc[idx]
    flag   = " ← CHANGED" if set(before) != set(after) else ""
    print(f"  Row {idx}: {before} → {after}{flag}")

# =============================================================================
# SECTION 9 — SAVE OUTPUTS
# Both files are used by all downstream blocks (Block 6 onwards).
# Block 6 should read df_cleaned and y_cleaned instead of df and y.
# =============================================================================

df_cleaned.to_csv(CLEANED_CSV, index=False)
np.save(CLEANED_NPY, y_cleaned)

print(f"\nSaved → {CLEANED_CSV}")
print(f"Saved → {CLEANED_NPY}")
print("\n=== BLOCK 5 COMPLETE ===")
print("Pass df_cleaned and y_cleaned into Block 6 (Split)")
