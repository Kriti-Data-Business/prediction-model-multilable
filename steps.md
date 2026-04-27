


**Multi-Label Complaint Tagger**
*Flat multi-label · ~96 tier_2 tags · DistilBERT + AsymmetricLoss*
*Blocks 0–15 · Auto-configured · Human-in-the-loop*

***

### The Core Problem

**What This Model Actually Does**

One complaint → multiple simultaneous `tier_2` tags (not a hierarchy)

- Input: `summary` + `Service` + `tier2_consolidated_description` + optional `keyword_group`
- Output: a **set** of tags from ~96 possible labels (e.g. *"Account breach"* + *"Consumer protection"* together)
- Challenge: severe imbalance, noisy human labels, dominant tags crowding out rare ones

**This is flat multi-label classification — not hierarchical.**

***

###  Why Not Simpler Models?

**Model Selection Reasoning**

| Approach | Why Rejected |
|---|---|
| OvR + Logistic Regression | 96 independent classifiers — no co-occurrence learning. "Account breach" and "Fraudulent transfer" always appear together but OvR never learns this |
| TF-IDF + BoW features | "I did NOT receive a refund" ≈ "I received a refund" — no word order or negation |
| `class_weight='balanced'` only | Still fails at 180× imbalance — rare tags stay at F1 ≈ 0 |
| Single threshold 0.5 | Dominant tags fire constantly, rare tags never fire |

**OvR's one valid use:** CleanLab probe in Block 5 — ~2 min, cached, generates cross-val probs for noise scoring. Not a final model.

***

###  Why DistilBERT?

**Why DistilBERT as the Encoder**

- **Shared encoder = joint learning:** all 96 tag classifiers share the same 66M-parameter backbone. The model learns that *"Account breach"* and *"Fraudulent transfer"* co-occur — something OvR structurally cannot do
- **Contextual embeddings:** negation, hedging, complaint tone all captured via attention — not bag-of-words
- **[CLS] pooling:** the single sentence-level vector feeds all 96 sigmoid outputs simultaneously
- **CPU-viable:** DistilBERT is 40% smaller than BERT-base with ~97% of performance — runnable without a GPU
- **Swap path:** one line change to `bert-base-uncased` if GPU becomes available

***

###  Why AsymmetricLoss?

**Why AsymmetricLoss Over BCELoss**

Standard BCELoss treats every tag × complaint pair equally. Your data is not equal.

- **Problem:** ~180× imbalance. BCELoss lets the model confidently predict 0 for rare tags and still achieve low loss — it never gets penalised enough
- **What AsymmetricLoss does:** applies a higher focusing penalty (`gamma_neg`) to easy negative examples (dominant tags the model over-predicts with high confidence)
- **`gamma_neg` auto-scales** from your imbalance ratio in Block 3 — no manual tuning
- **`gamma_pos=1` stays gentle** on rare positive examples — doesn't over-punish misses on tags with very few samples

***

###  Block 0: Smart Sampling

**BLOCK 0 — Smart Stratified Sampling**

*Previous approach:* Full dataset (~20k rows) → ~33 hrs per epoch on CPU. Unusable.

- **Phase 1:** Guaranteed quota of 10–20 cases per `keyword_group` — rare keywords can't be excluded
- **Phase 2:** Fill to 3,500 total via inverse-frequency weighted sampling — common keywords get capped
- Output: `saved_data/shortlisted_3500.csv` — reproducible, keyword-balanced

***

###  Block 3: Auto-Diagnose

**BLOCK 3 — Auto-Diagnose (No Hardcoded Hyperparameters)**

*Previous approach:* Fixed `EPOCHS=8`, `BATCH_SIZE=32`, `BASE_SMOOTH=0.05` regardless of data.

Reads your actual data at runtime and computes everything:

- Imbalance ratio → `GAMMA_NEG`, `KW_DROPOUT`, `DO_UNDERSAMPLE`
- Avg CleanLab quality score → `BASE_SMOOTH`
- Dataset size + median text length → `EPOCHS`, `BATCH_SIZE`, `MAX_LEN`

***

###  Block 5: CleanLab

**BLOCK 5 — CleanLab Noise Detection**

*Previous approach:* OvR used as the production model. Macro F1 ceiling ~0.28–0.42 — structural, not tunable.*

OvR is repurposed as a fast probe only:

- TF-IDF (15k features) + 5-fold cross-val → generates out-of-fold probabilities in ~2 min
- `multilabel_label_quality_scores()` → per-row `noise_score` (0 = clean, 1 = noisy)
- High `noise_score` rows get more label smoothing in Block 8
- **Cached to `checkpoints/cleanlab_scores.csv`** — never reruns unless deleted

Why it matters: one mislabelled complaint flows gradient errors into all 96 tag classifiers simultaneously via the shared BERT encoder.

***

### Block 6: Split Before Augmenting

**BLOCK 6 — Split Before Augmentation**

*Previous approach:* Augmented training data first, then split — validation set contained augmented copies of training rows. Inflated F1.*

- 80% train / 10% val / 10% test — fixed before any augmentation
- All three splits saved to disk (`train_split.csv`, `val_split.csv`, `test_split.csv`)
- CleanLab cache reused across retraining runs

***

###Block 7: Three-Layer Imbalance Handling

**BLOCK 7 — Imbalance Handling (3 Layers)**

*Previous approach:* `class_weight='balanced'` alone. F1 ≈ 0 on rare tags at 180× ratio.*

Applied in order on training set only:

- **Layer 1 — Augment rare tags:** word swap / drop / repeat on complaints containing tags below `RARE_THRESHOLD`
- **Layer 2 — Undersample dominant-only:** remove 50% of rows where the only tag is the top dominant tag
- **Layer 3 — WeightedRandomSampler:** per-row weights = inverse tag frequency → rare-tag rows seen more per epoch

***

###  Block 8: Keyword Dropout

**BLOCK 8 — Dataset & Keyword Dropout**

*Previous approach:* `keyword_group` always included → model shortcut on human-assigned keywords instead of reading the complaint text.*

- `build_text` always includes: `Service` + `summary` + `tier2_consolidated_description`
- `keyword_group` masked with probability `KW_DROPOUT` during training only
- `KW_DROPOUT` auto-set from imbalance ratio (range 0.30–0.55)
- Adaptive label smoothing: each row's `ε = BASE_SMOOTH + noise_score × 0.10` — noisier rows get softer targets

***

###  Block 11: Per-Tag Thresholds

**BLOCK 11 — Per-Tag Threshold Tuning**

*Previous approach:* Single threshold 0.5 — dominant tags fire on almost everything, rare tags never fire.*

- Grid search per tag: 0.10 → 0.90 (step 0.05) on validation set
- Dominant tags (`freq > DOMINANT_FREQ`) → threshold forced ≥ 0.65
- Rare tags (`freq < RARE_FREQ`) → threshold forced ≤ 0.35
- Saved to `saved_data/tag_thresholds.npy` — used at inference
 
***

### Block 13–14: Human Review Loop

**BLOCKS 13–14 — Human Review + Feedback Loop**

Block 13 exports three batches to `human_review/complaints_for_review.csv`:

- **Uncertain:** high entropy predictions — model is unsure across multiple tags
- **Rare-tag predicted:** verify the model got rare tags right before trusting them
- **High-confidence:** spot-check for systematic overconfidence / false positives

Block 14 merges corrections back into `train_split.csv` → retrain from Block 2. CleanLab cache is reused — no re-running the 2-min probe.

***

### Block 16: Backtracking

**BLOCK 16 — Rich Prediction Output + Backtracking**

For every test complaint:

- Top 10 predicted tags with % confidence bars
- CORRECT / MISSED / FALSE ALARM status per tag
- For each mismatch: top 8 input tokens (from [CLS] attention rollout) that most influenced the wrong prediction
- Full results saved as CSV + HTML report

Used to diagnose *why* specific tags fail — feeds directly into Block 13 review prioritisation.

# In Block 12 — after classification_report
```
per_tag_f1 = f1_score(tl_h, preds, average=None, zero_division=0)
thr_df['test_f1'] = per_tag_f1

# Bucket your tags
zero_f1    = thr_df[thr_df['test_f1'] == 0]
low_f1     = thr_df[(thr_df['test_f1'] > 0) & (thr_df['test_f1'] < 0.4)]
decent_f1  = thr_df[thr_df['test_f1'] >= 0.4]

print(f"Tags with F1 = 0.00 : {len(zero_f1)}  ← fix these first")
print(f"Tags with F1 < 0.40 : {len(low_f1)}")
print(f"Tags with F1 >= 0.40: {len(decent_f1)}")
print(zero_f1[['tag','train_freq_%','threshold','test_f1']].to_string())
```
***

