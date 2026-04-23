# prediction-model-multilable


Settings That Auto-Scale

| Setting         | Formula                              | Simulated Data | Severe Imbalance Example    |
| --------------- | ------------------------------------ | -------------- | --------------------------- |
| MIN_TAG_SAMPLES | max(3, dataset_size × 0.005)         | 5              | 15 if you have 3000+ rows   |
| KW_DROPOUT      | 0.20 + imbalance_ratio / 200         | 0.35           | 0.50 if 60x imbalance       |
| RARE_THRESHOLD  | dataset_size × 0.03                  | 20             | 90 if 3000 rows             |
| GAMMA_NEG (ASL) | 2 + imbalance_ratio / 30             | 4              | 6 if severely imbalanced    |
| EPOCHS          | 20000 / dataset_size + 3             | 6              | 10 if only 1000 rows        |
| BATCH_SIZE      | Based on dataset size                | 16             | 8 if tiny dataset           |
| MAX_LEN         | From median summary length           | 384            | 512 if very long complaints |
| BASE_SMOOTH     | 0.02 + (1 - cleanlab_quality) × 0.20 | 0.03           | 0.12 if quality=0.5         |
| DO_AUGMENT      | Only if rare tags exist              | True/False     | Auto                        |
| DO_UNDERSAMPLE  | Only if dominant tags > 50%          | False          | True if real imbalance      |

=== DATA DIAGNOSIS ===
Total complaints:        1200
Unique tier_2 tags:      96
Imbalance ratio:         180x        ← triggers GAMMA_NEG=6, KW_DROPOUT=0.50
Tags with < 5 samples:  34           ← MIN_TAG_SAMPLES=6, those 34 dropped
CleanLab avg quality:   0.71         ← BASE_SMOOTH=0.08 (much more smoothing)
CleanLab pct flagged:   82%

=== AUTO-COMPUTED SETTINGS ===
GAMMA_NEG:  6    (scaled up from high imbalance)
KW_DROPOUT: 0.50 (scaled up — humans clearly relied on keywords)
BASE_SMOOTH: 0.08 (scaled up — your labels are genuinely noisy)
DO_AUGMENT: True
DO_UNDERSAMPLE: True
