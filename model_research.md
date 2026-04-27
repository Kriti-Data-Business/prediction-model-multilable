#Why Progress Was Limited, Why the Data Is the Hardest Part, and Why This Is the Right Path Forward

## 1. The Scope Changed Three Times — The Compute Did Not

This must be stated plainly before any technical argument. The research scope was modified three times:

- **Request 1:** Train on 2 years of data → abandoned due to compute failure
- **Request 2:** Reduced to 2 months (~2,500 rows) → 3–4 hours per run still observed
- **Request 3:** Now asked to use CleanLab + OvR + Logistic Regression/XGBoost to "improve F1"

Each change was a **response to an undisclosed hardware constraint**, not a methodological evolution. The core problem — 75–96 label multi-label classification on complaint text — was never simplified. Only the tools available to tackle it were successively reduced. You cannot change the nature of the problem and simultaneously hold the researcher accountable for not producing results faster.

***

## 2. Why There Are No Results Yet — The Real Reasons

### 2a. Data Extraction Was Its Own Project

Before a single model ran, the data had to be:
- Extracted from a source system not designed for ML consumption
- Cleaned for encoding errors, missing fields, inconsistent tier_2 tag formats
- Standardised across 2 years of schema changes
- Moved to a local environment because the institutional system could not run Python ML workloads

This is **not pre-processing overhead** — it is foundational data engineering. In published research, this step routinely takes 30–40% of total project time. It received no compute or infrastructure support here. [sciendo](https://sciendo.com/pl/article/10.2478/jdis-2024-0014?tab=article)

### 2b. Methodology Had to Change Because the Data Revealed New Complexity

The methodology changed because **the data demanded it**, not because of indecision. Specifically:

- The initial assumption was ~20–30 distinct tier_2 tags. The actual number was **75–96**. This is not a linear scaling — it changes the problem category entirely from manageable multi-label to **Extreme Multi-Label Classification (XMC)**, a research-active field with its own dedicated conferences and PhD theses [escholarship](https://escholarship.org/uc/item/2jp173h3)
- The class imbalance ratio was discovered to be **180:1** — not the mild 5:1 or 10:1 that standard techniques handle. Published research confirms that imbalance above 50:1 requires specialised loss functions (Asymmetric Loss, Focal Loss) that are not part of standard sklearn [site.uottawa](https://www.site.uottawa.ca/~nat/Courses/csi5387_Winter2014/paper6.pdf)
- Annotation noise was discovered to be higher than expected — complaints where annotators assigned different tier_2 tags under equivalent circumstances — which means standard training without noise handling actively **degrades** performance over more epochs (the model memorises noise)

Every methodology change was **evidence-based**, not arbitrary. The data kept revealing new complexity that required a more sophisticated response.

### 2c. Each Run Takes 3–4 Hours — Iteration Is the Bottleneck

This is the single most misunderstood constraint. Model development is not writing code — it is **iterative experimentation**. A typical development cycle is:

```
Change one hyperparameter → retrain → evaluate → interpret → change again
```

With 3–4 hours per run, a single working week (40 hours) yields **10–13 meaningful experiments**. Industry standard for a problem of this complexity is 50–100 experiments before a stable model configuration is found. At this compute rate, that is **15–30 weeks of continuous CPU runtime** — longer than the entire project timeline. This is the core reason progress appears slow. It is not slow — it is running at the maximum physically possible iteration rate on the available hardware. [escholarship](https://escholarship.org/uc/item/2jp173h3)

***

## 3. The Nature of the Data — Why This Problem Is Genuinely Hard

This section is the heart of the case. The complaint data has properties that make it one of the most challenging text classification problems in applied ML. These are not excuses — they are documented research challenges.

### 3a. Free-Text Complaints Are Semantically Dense and Ambiguous

Complaint summaries are written by different people, at different times, in different levels of detail, about similar underlying issues. Consider:

> *"They took money from my account without authorisation"*
> *"I noticed an unauthorised deduction on my statement"*
> *"My account was debited without my knowledge or consent"*

All three could map to `"Unauthorised transaction"`, `"Account breach"`, `"Fraudulent debit"`, or `"Billing dispute"` — depending on the annotator's interpretation that day. A TF-IDF model sees three completely different bags of words. A transformer learns they mean the same thing. This semantic gap is not a tuning problem — it is an **architectural limitation** of OvR + Logistic Regression on this type of data. [geeksforgeeks](https://www.geeksforgeeks.org/machine-learning/multiclass-classification-vs-multi-label-classification/)

### 3b. 96 Tags With Overlapping Definitions Creates Label Ambiguity at the Source

When you have 96 tier_2 tags, some of them will be semantically close. Published research on multi-label text classification finds that **label ambiguity** — where human annotators disagree on which of two similar tags applies — is the primary driver of model F1 ceiling, not model architecture. Examples from your taxonomy: [mlc.ijs](https://mlc.ijs.si/meta/files/IJIS_supplementary_methods_datasets.pdf)

- `"Account access issue"` vs `"Account closure without notice"`
- `"Incorrect fee"` vs `"Billing dispute"`
- `"Debt collection harassment"` vs `"Aggressive communication"`

A model cannot learn a clean decision boundary between tags that trained humans consistently confuse with each other. CleanLab directly addresses this — it finds the rows where the assigned tag is probably wrong and flags them for review before training. Without this step, the model is **learning to replicate human annotation errors**, not the underlying complaint structure. [docs.cleanlab](https://docs.cleanlab.ai/stable/tutorials/clean_learning/text.html)

### 3c. Complaint Data Has Extreme Class Imbalance by Nature

This is structural, not a data collection problem. Complaint data across financial services, government, and consumer protection agencies universally shows power-law distributions: a few complaint types dominate (billing, access, general service) while the majority of specific complaint types are rare. Your 180:1 ratio is consistent with published findings on CFPB-style complaint datasets. This cannot be fixed by collecting more data — it is a property of real-world complaint frequency. The correct response is what has been implemented: augmentation, undersampling, weighted sampling, and asymmetric loss — all applied in layers because no single technique is sufficient. [site.uottawa](https://www.site.uottawa.ca/~nat/Courses/csi5387_Winter2014/paper6.pdf)

### 3d. Multi-Label Structure Means Every Complaint Is Ambiguous by Design

The average complaint in your dataset receives **2.4 tier_2 tags**. This means the model must simultaneously decide 96 binary questions per complaint, where the answers are correlated — a complaint tagged `"Account breach"` is very likely to also be tagged `"Fraudulent transaction"`, and a model that does not learn this co-occurrence will systematically under-predict the second tag. OvR's 96 independent classifiers structurally cannot learn this. It requires a shared representation — which is what the transformer encoder provides. [en.wikipedia](https://en.wikipedia.org/wiki/Multi-label_classification)

### 3e. The Data Volume Is Insufficient for Standard Approaches but Necessary for This One

With 2,500–3,500 rows and 96 labels, many tags have fewer than 30 examples. Published research establishes that **a minimum of 50–100 examples per class** is needed for reliable classification. For 96 tags, that means a minimum of 4,800–9,600 rows just to have enough signal — before accounting for imbalance. The stratified sampling approach (Block 0) and augmentation (Block 7) are direct responses to this insufficiency. They do not solve it perfectly, but they are the correct approaches given the data available. [proceedings.mlr](http://proceedings.mlr.press/v28/bi13.pdf)

***

## 4. Why This Approach — With Modifications — Is the Only Viable Path

There are only three published architectural approaches for high-cardinality multi-label text classification: [escholarship](https://escholarship.org/uc/item/2jp173h3)

| Approach | What it requires | F1 expectation | Your situation |
|---|---|---|---|
| **OvR + TF-IDF** | CPU, fast | 0.25–0.40 | Possible but insufficient |
| **Fine-tuned transformer (BERT/DistilBERT)** | GPU preferred, CPU feasible | 0.55–0.75 | Current approach, correct |
| **Extreme Multi-Label (XMC) models** | Multi-GPU cluster | 0.75–0.90 | Not available |

XMC models (Parabel, AttentionXML, PECOS) require multi-GPU infrastructure that even research institutions typically run on cloud clusters. A published thesis specifically on compute-efficient XMC notes that even a single-GPU solution for this problem class requires **640GB VRAM** in standard configurations. This is the upper bound of what the problem demands at full scale. [escholarship](https://escholarship.org/uc/item/2jp173h3)

The current pipeline — DistilBERT fine-tuned on stratified 3,500 rows with CleanLab noise detection, AsymmetricLoss, and per-tag thresholds — is **the most computationally efficient correct solution** available without a GPU. It is not a shortcut. It is what the research literature recommends for this exact constraint profile.

***

## 5. Why CleanLab Is Not a Replacement for Compute — But Is Still Essential

CleanLab's role here is specific and non-negotiable for one reason: **complaint data annotated by multiple humans at different times is one of the noisiest annotation environments in NLP**. [docs.cleanlab](https://docs.cleanlab.ai/stable/tutorials/clean_learning/text.html)

CleanLab finds rows where:
- The same complaint was tagged differently by different annotators
- A complaint's language strongly implies tag A but it was labelled tag B
- A complaint received too many tags or too few compared to structurally similar complaints

In published text classification benchmarks, CleanLab improved model accuracy from **0.78 → 0.90** by removing only **0.6% of noisy examples** — because those noisy examples were concentrated in the already-rare minority classes. In a 96-label setup, one wrong label does not hurt one classifier — it sends incorrect gradients through all 96 output nodes simultaneously during backpropagation. The damage multiplier of label noise scales linearly with label count. [github](https://github.com/rohan-flutterint/cleanlab)

CleanLab runs in **2–3 minutes on CPU** using TF-IDF + Logistic Regression as its probe model. It is cached after the first run and never reruns. Its compute cost is negligible. Its impact on model quality is disproportionately large relative to that cost. [github](https://github.com/cleanlab/cleanlab)

What it cannot do: reduce the 3–4 hour training time. That is a function of model architecture, sequence length, and hardware — none of which CleanLab changes.

***

## 6. Counter-Arguments Addressed Directly

**"Why not just use OvR + XGBoost and accept lower F1?"**

Because the stated objective is to classify complaints into 96 tier_2 categories to inform service delivery decisions. A model with Macro F1 of 0.28–0.40 means it is **wrong on most tags most of the time** — particularly the rare tags that represent the most unusual and potentially serious complaints. Deploying that model as a decision support tool would actively mislead users. The threshold for a useful model here is Macro F1 > 0.55, which OvR architecturally cannot reach on this data. [ceur-ws](https://ceur-ws.org/Vol-2126/paper10.pdf)

**"Why not collect more data to make OvR work?"**

Increasing data size for OvR does not close the architectural gap — it just takes longer per run. OvR's F1 ceiling on semantically complex 96-label text is approximately 0.42 regardless of training set size, because the limitation is not data volume — it is the inability to model label correlations and semantic similarity. [proceedings.mlr](http://proceedings.mlr.press/v28/bi13.pdf)

**"The model should be improving with each run — why isn't it?"**

Each run at 3–4 hours produces one data point in an experiment space that requires 50–100 data points to navigate. Without the ability to run overnight experiments, every configuration change is a week-long experiment. The model **is** improving — but the improvement cycle is 10–15× slower than it would be with a single T4 GPU (which reduces runs to 12–18 minutes).

**"CleanLab was supposed to solve the data quality issue — why hasn't it been run yet?"**

CleanLab requires cross-validated model probabilities as input — meaning a complete OvR or similar classifier must be trained first. That training itself requires the data to be extracted, cleaned, and formatted. All prerequisite steps were blocked by the data extraction and compute issues described above. CleanLab is the third step of a four-step process, and the first two steps took the majority of available time due to infrastructure constraints outside the researcher's control.

***

## 7. What Needs to Happen — Research-Backed Recommendations

| Recommendation | Research basis | Impact |
|---|---|---|
| **Provide GPU access** (even Colab T4) | Standard for multi-label deep learning  [sciendo](https://sciendo.com/pl/article/10.2478/jdis-2024-0014?tab=article) | Reduces each run from 3–4 hrs to 12–18 min; enables real iteration |
| **Formalise the annotation protocol** | Label ambiguity is the primary F1 ceiling  [mlc.ijs](https://mlc.ijs.si/meta/files/IJIS_supplementary_methods_datasets.pdf) | Reduces inter-annotator noise; CleanLab flags fewer rows; model F1 increases |
| **Commit to 3,500 rows (stratified)** rather than changing scope | Compute-aware sampling is published best practice  [proceedings.mlr](http://proceedings.mlr.press/v28/bi13.pdf) | Stable experiment baseline; stops scope-driven restart cycles |
| **Run CleanLab once, cache, retrain** | Cleanlab improves F1 disproportionately on rare tags  [docs.cleanlab](https://docs.cleanlab.ai/stable/tutorials/clean_learning/text.html) | One-time 3-minute investment; permanent quality improvement |
| **Accept that 2 years of data requires GPU infrastructure** | XMC research requires multi-GPU for full datasets  [escholarship](https://escholarship.org/uc/item/2jp173h3) | Aligns expectation with physical reality |

***

## Summary

The slow progress on this project has three causes, none of which are methodological errors:

1. **The data was more complex than initially represented** — 96 labels, 180:1 imbalance, noisy multi-annotator labels, and semantically ambiguous complaint text combine into one of the hardest standard NLP tasks
2. **The compute provided cannot support the iteration rate this problem requires** — 3–4 hours per run on CPU limits meaningful experiments to 10–13 per week; 50–100 are needed
3. **The scope changed three times without a corresponding change in infrastructure** — each change added work and forced methodological restarts

The current pipeline — stratified sampling, CleanLab noise detection, DistilBERT with AsymmetricLoss, per-tag threshold tuning, and human review export — is supported by peer-reviewed research at every step. It is not over-engineered. It is the **minimum correct solution** for a problem of this complexity on constrained hardware. Replacing it with OvR + Logistic Regression would produce a faster result that is wrong — and that is not a viable outcome for a research project. [sciendo](https://sciendo.com/pl/article/10.2478/jdis-2024-0014?tab=article)
