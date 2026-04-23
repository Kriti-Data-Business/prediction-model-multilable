### CleanLab doesn't help with class imbalance  it only finds label noise
reference : https://www.nature.com/articles/s41598-025-05791-7

For  severe tag imbalance, SMOTE-ENN (oversampling minority + removing borderline majority) is the state-of-the-art combination
```
from imblearn.combine import SMOTEENN
from sklearn.feature_extraction.text import TfidfVectorizer

# Run on TF-IDF features first (not raw BERT — too slow for resampling)
tfidf = TfidfVectorizer(max_features=5000)
X_tfidf = tfidf.fit_transform(df_tr['summary'].fillna(''))

# Works per-tag: apply to each rare tag binary label
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours

for i, tag in enumerate(valid_tags):
    if y_tr[:, i].sum() < RARE_THRESHOLD:
        smote_enn = SMOTEENN(random_state=42)
        X_res, y_res = smote_enn.fit_resample(X_tfidf.toarray(), y_tr[:, i])
        # Use resampled indices to guide your WeightedRandomSampler
```
Why it beats pure augmentation:  word-swap augmentation creates near-duplicate text. SMOTE creates synthetic feature-space samples that genuinely diversify the minority class.

### Active Learning with Label Studio (Best for Budget Constraints)
Instead of cleaning all labels at once, only re-annotate the complaints your model is most uncertain about. This gives you maximum quality improvement per hour of human review:
```
# After your first training run:
# Entropy = uncertainty score per complaint
def prediction_entropy(probs):
    p = np.clip(probs, 1e-9, 1-1e-9)
    return -(p * np.log(p) + (1-p) * np.log(1-p)).mean(axis=1)

entropy = prediction_entropy(val_probs)  # shape: (n_samples,)

# Export the 200 most uncertain complaints for human re-review
uncertain_idx = np.argsort(entropy)[-200:]
```
#### Why it beats CleanLab: You fix the rows that actually hurt your model, not just statistically suspicious ones
df.iloc[uncertain_idx][['case_reference', 'summary', 'tags_parsed']]\
  .to_csv('needs_review.csv', index=False)
# → Give this to a domain expert or run through LLM re-labeller
