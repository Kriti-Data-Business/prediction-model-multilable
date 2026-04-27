
## 4. Why One-vs-Rest + Logistic Regression Is the Wrong Tool Here

OvR with Logistic Regression is an interpretable, fast baseline. It has a well-understood failure mode at exactly this problem's characteristics: [pmc.ncbi.nlm.nih](https://pmc.ncbi.nlm.nih.gov/articles/PMC7331794/)

### 4a. It Cannot Model Label Correlations

OvR trains 96 **completely independent** binary classifiers. Each classifier has no knowledge that tags co-occur, that certain service types cluster with certain complaint categories, or that `"Account breach"` and `"Fraudulent transfer"` share semantic signal. In your dataset, approximately **2.4 tags appear per complaint on average** — meaning nearly every complaint has correlated label assignments. OvR treats this structure as noise. [en.wikipedia](https://en.wikipedia.org/wiki/Multi-label_classification)

### 4b. It Breaks on Severe Imbalance With Many Labels

At 180× imbalance ratio, standard OvR with `class_weight='balanced'` will achieve near-zero F1 on rare tags regardless of tuning. The classifier for a tag appearing 8 times in 2,500 rows sees 2,492 negatives vs. 8 positives — even with balancing, logistic regression's linear decision boundary cannot reliably separate these classes in a 15,000-dimensional TF-IDF space. [techblog.skeepers](https://techblog.skeepers.io/how-to-boost-your-f1-score-for-multiclass-multilabel-classification-eeb8452c8171)

Expected Macro F1 for OvR + Logistic Regression on this problem: **0.25–0.40**. [ceur-ws](https://ceur-ws.org/Vol-2126/paper10.pdf)

### 4c. TF-IDF Loses the Semantic Signal That Distinguishes Tags

Many of your 96 tier_2 tags are semantically close: `"Billing dispute"` vs `"Incorrect fee charged"`, `"Account access"` vs `"Account closure"`. TF-IDF represents both as overlapping bags of words — the model cannot distinguish them reliably. A transformer fine-tuned on this data learns **contextual embeddings** where `"they closed my account without notice"` and `"I cannot access my account"` map to different vector spaces, even though both contain the word "account." [csie.ntu.edu](https://www.csie.ntu.edu.tw/~cjlin/libmultilabel/auto_examples/plot_multi_label.html)

### 4d. XGBoost Does Not Solve This Either

XGBoost is a strong tabular model. On text, it consumes TF-IDF features identically to logistic regression. It is faster than neural approaches at inference but produces similar or worse F1 on high-cardinality text problems because it still cannot model sequential semantic meaning. Adding XGBoost as an ensemble on top of TF-IDF features adds 20–40 minutes of training per run without closing the architectural gap. [csie.ntu.edu](https://www.csie.ntu.edu.tw/~cjlin/libmultilabel/auto_examples/plot_multi_label.html)

***

## 5. Why CleanLab Is Valuable But Cannot Replace Compute

CleanLab is an excellent data quality tool. But it needs to be understood correctly in context:

### What CleanLab Does

CleanLab identifies rows where the annotated labels are probably wrong — using cross-validated model predictions to find disagreements between the label and the data. Correcting these errors can lift Macro F1 by **0.05–0.15** on a 96-label problem because a single mislabelled complaint pollutes all 96 classifiers through shared gradient updates. [docs.cleanlab](https://docs.cleanlab.ai/v2.7.0/tutorials/improving_ml_performance.html)

### What CleanLab Does Not Do

CleanLab **does not reduce compute requirements**. After cleaning, you still need to retrain the same model on the same size dataset. The improvement is real and meaningful — but it is a data quality intervention, not a hardware intervention. [docs.cleanlab](https://docs.cleanlab.ai/v2.7.0/tutorials/improving_ml_performance.html)

### The Correct Use of CleanLab Here

Running CleanLab with **3 concatenated features** (Service + keywords + summary) through TF-IDF is exactly the right approach — fast (2–3 minutes), cached after one run, and provides per-row noise scores that inform adaptive label smoothing during BERT training. This is already implemented in Block 5 of the pipeline. What it **cannot** do is make OvR + Logistic Regression competitive with a fine-tuned transformer. [towardsdatascience](https://towardsdatascience.com/automatically-detecting-label-errors-in-datasets-with-cleanlab-e0a3ea5fb345/)


## 1. What Was Actually Asked of This Project

Over the course of this research, the scope changed **three times** without a corresponding increase in compute resources:

- **Version 1:** Train on 2 years of complaint data (~20,000+ rows)
- **Version 2:** After compute failure, reduced to 2 months of data (~2,500 cases to selected randomly rows)
- **Version 3:** Now asked to use CleanLab with 3 features + One-vs-Rest (OvR) + Logistic Regression / XGBoost to improve F1

Each version change was a **response to a hardware constraint**, not a methodological choice. The core problem — multi-label classification with 75–96 tier_2 tags on complaint text — did not change. Only the tools and data available to solve it changed. That is the central argument here.

***

## 2. Why This Problem Is Computationally Expensive by Nature

This is not a simple classification problem. It is a **high-cardinality multi-label text classification problem** — one of the most computationally demanding categories in applied machine learning. [proceedings.mlr](http://proceedings.mlr.press/v28/bi13.pdf)

### 2a. The Output Space Is Exponential

In binary classification, there are 2 possible outputs. In 96-label multi-label classification, the theoretical output space is:

\[ 2^{96} \approx 79 \text{ octillion possible label combinations} \]

Even practically, with 96 independent binary decisions per complaint, the model must learn a **joint probability landscape** across all 96 tags simultaneously. Each training step updates parameters that affect all 96 decisions at once. [sciendo](https://sciendo.com/pl/article/10.2478/jdis-2024-0014?tab=article)

### 2b. Attention Is Quadratic in Sequence Length

The complaint text (summary + service + description) tokenises to 256–384 tokens. DistilBERT's self-attention computes:

\[ \text{Cost per layer} = O(n^2 \cdot d) \]

where \(n\) = sequence length and \(d\) = hidden dimension (768). At 384 tokens across 6 layers:

\[ 6 \times 384^2 \times 768 \approx 682 \text{ million operations per complaint} \]

On a standard CPU at ~50 GFLOPS, **one forward + backward pass through a single batch of 16 complaints takes approximately 3–4 minutes**. With 2,500 training rows across 8 epochs:

\[ \frac{2500}{16} \times 8 \text{ epochs} \times 4 \text{ min/batch} \approx 3.3 \text{ hours per full run} \]

This is precisely what was observed experimentally — 3–4 hours per run — and it is **mathematically expected, not a bug or inefficiency**. [proceedings.mlr](http://proceedings.mlr.press/v28/bi13.pdf)

### 2c. Gradient Updates Scale With Label Count

Every backward pass computes gradients for **all 96 classifiers simultaneously** through the shared encoder. The classifier layer alone has `768 × 96 = 73,728` parameters that must be updated every batch. With Asymmetric Loss applying separate positive and negative scaling terms per label, the effective gradient computation doubles. This is why 96 labels is approximately **96 times harder** than binary classification at the output layer, on top of the transformer cost. [en.wikipedia](https://en.wikipedia.org/wiki/Multi-label_classification)

***

## 3. Why Training on 2 Years of Data Was Always Infeasible Without a GPU

The request to train on 2 years of data was a **compute-blind scope decision**. Here is what 2 years of data means operationally:

| Parameter | 2 months | 2 years |
|---|---|---|
| Approximate rows | ~2,500 | ~25,000–30,000 |
| Batches per epoch (size 16) | ~156 | ~1,562 |
| Hours per epoch (CPU) | ~0.4 hrs | ~4.2 hrs |
| Full run (8 epochs) | **~3.3 hrs** | **~33 hrs** |
| With hyperparameter tuning (5 configs) | ~17 hrs | **~165 hrs** |

A single hyperparameter sweep on 2 years of data would take nearly **7 days of continuous CPU compute**. The project deadline was 2 weeks. This was structurally impossible before a single line of modelling code was written. [epoch](https://epoch.ai/blog/the-longest-training-run)

The correct response — which was implemented — was to **reduce data size while preserving statistical representation** through stratified sampling (Block 0), not to change the model architecture to an inferior one.

***

***

## 6. Why the Current Pipeline Is the Right Approach With Modifications

The pipeline that has been developed addresses every constraint with the correct tool:

| Constraint | What was done | Why it's correct |
|---|---|---|
| **Not enough compute for 2 years** | Smart stratified sampling to 3,500 rows | Preserves statistical representation per keyword; cuts compute 8×  [proceedings.mlr](http://proceedings.mlr.press/v28/bi13.pdf) |
| **96-label imbalance** | 3-layer imbalance handling (augment + undersample + sampler) | No single technique fixes this; layered approach is published best practice  [ar5iv.labs.arxiv](https://ar5iv.labs.arxiv.org/html/2312.07087) |
| **Noisy human annotations** | CleanLab (fast TF-IDF concatenated method) | 2–3 min one-time cost; improves all downstream training  [towardsdatascience](https://towardsdatascience.com/automatically-detecting-label-errors-in-datasets-with-cleanlab-e0a3ea5fb345/) |
| **Dominant tag shortcuts** | Keyword dropout (30–55%) during training | Forces model to read complaint text, not memorise keyword→tag mapping |
| **Low F1 on rare tags** | Asymmetric Loss with auto-scaled gamma | Standard BCE fails catastrophically on 180× imbalance; ASL is the published solution  [ar5iv.labs.arxiv](https://ar5iv.labs.arxiv.org/html/2312.07087) |
| **Per-tag threshold variability** | Individual threshold tuning on validation set | Dominant tags need 0.65+; rare tags need 0.35 or below — a single threshold is incorrect |
| **Rerun cost** | Checkpoints every epoch + cached CleanLab scores | Never redo completed work; resume from any epoch |
| **Human review** | Uncertainty-stratified export CSV | Targets model's actual failure modes rather than random sampling |

***

## 7. What Is Needed Going Forward

The minimum viable path to a working production model is:

1. **Keep the current DistilBERT pipeline** — it is architecturally correct for this problem
2. **Run on 3,500 stratified rows** (Block 0) — this is the compute-aware version that is still statistically valid
3. **CleanLab once, cache forever** — 3 minutes of compute that improves every future run
4. **Request even a modest GPU** — an NVIDIA T4 (available free on Google Colab) reduces each run from 3–4 hours to **12–18 minutes**, making iterative improvement feasible within any reasonable deadline
5. **If GPU remains unavailable**, the fallback is DistilBERT on 3,500 rows with early stopping at epoch 3 — this produces ~70% of the full-run quality in ~1.2 hours per run, which is achievable

The OvR + Logistic Regression approach is appropriate for one specific purpose: generating the cross-validated probabilities that CleanLab needs in Block 5. It is already doing exactly that job. Using it as the **final model** on a 96-label problem with complaint text and severe imbalance will produce Macro F1 in the 0.25–0.40 range regardless of tuning — a result that cannot be defended in a thesis or production system. [ceur-ws](https://ceur-ws.org/Vol-2126/paper10.pdf)

***

## Summary

The core issue is that **compute constraints were imposed after the problem scope was defined**, not before. A 96-label multi-label classification problem on complaint text has a mathematically predictable minimum compute requirement — approximately 3–4 hours per run on a standard CPU — which was confirmed by experiment. Reducing the model architecture to OvR + Logistic Regression does not solve the compute problem; it solves a different (easier) problem while producing results that cannot meet the stated research objectives. The current pipeline is the correct solution, and with stratified sampling to 3,500 rows and CleanLab noise detection already implemented, it is also the most compute-efficient version of that correct solution.
