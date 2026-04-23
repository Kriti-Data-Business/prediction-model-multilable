
# ============================================================
# DYNAMIC SELF-ADAPTING COMPLAINT TAGGER PIPELINE
# Automatically reads your real data and adjusts all settings
# based on actual imbalance, noise, and tag distribution
# ============================================================

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random, ast, os
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from tqdm import tqdm

# ============================================================
# STEP 0 — DYNAMIC CONFIG (auto-set from your data)
# Only hardcode model name and file path
# ============================================================
MODEL_NAME = 'distilbert-base-uncased'  # swap to bert-base-uncased if GPU
DATA_FILE  = 'cfpb_complaints_simulated.csv'   # ← your file

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}  |  Model: {MODEL_NAME}")
print("=" * 60)
print("DYNAMIC PIPELINE — all settings auto-computed from your data")
print("=" * 60)

# ============================================================
# STEP 1 — LOAD DATA
# ============================================================
df = pd.read_csv(DATA_FILE)
print(f"\nLoaded: {len(df)} rows, {df.shape[1]} columns")

# ============================================================
# STEP 2 — PARSE TAGS AND KEYWORD GROUPS
# ============================================================
def parse_tags(val):
    if isinstance(val, list):
        return [t.strip() for t in val if str(t).strip()]
    if isinstance(val, str):
        val = val.strip()
        if val.startswith('['):
            try: return [t.strip() for t in ast.literal_eval(val) if str(t).strip()]
            except: pass
        return [t.strip() for t in val.split(',') if t.strip()]
    return []

def parse_list_str(val):
    if isinstance(val, list): return val
    if isinstance(val, str) and val.startswith('['):
        try: return ast.literal_eval(val)
        except: pass
    return [val] if isinstance(val, str) and val.strip() else []

df['tags_parsed']          = df['tier_2'].apply(parse_tags)
df['keyword_group_parsed'] = df['keyword_group'].apply(parse_list_str)
df = df[df['tags_parsed'].apply(len) > 0].reset_index(drop=True)
print(f"After dropping untagged rows: {len(df)}")

# ============================================================
# STEP 3 — BINARIZE LABELS
# ============================================================
mlb = MultiLabelBinarizer()
y   = mlb.fit_transform(df['tags_parsed'])
tag_counts_s = pd.Series(y.sum(axis=0), index=mlb.classes_).sort_values(ascending=False)

# ============================================================
# STEP 4 — AUTO-DIAGNOSE DATA QUALITY
# Sets all downstream hyperparameters dynamically
# ============================================================
n_total       = len(df)
n_tags_raw    = len(mlb.classes_)
tags_per_row  = y.sum(axis=1)
max_count     = tag_counts_s.max()
min_count     = tag_counts_s.min()
imbalance_ratio = max_count / max(min_count, 1)

# --- CleanLab columns ---
has_cleanlab = 'cleanlab_avg_quality' in df.columns
if has_cleanlab:
    avg_quality  = df['cleanlab_avg_quality'].fillna(0.9).mean()
    pct_flagged  = df['has_label_issue'].astype(str).str.upper().eq('TRUE').mean()
    min_quality  = df['cleanlab_min_quality'].fillna(0.9).min()
else:
    avg_quality, pct_flagged, min_quality = 0.9, 0.0, 0.9

print(f"\n=== DATA DIAGNOSIS ===")
print(f"Total complaints:        {n_total}")
print(f"Unique tier_2 tags:      {n_tags_raw}")
print(f"Avg tags per complaint:  {tags_per_row.mean():.2f}")
print(f"Max tags per complaint:  {int(tags_per_row.max())}")
print(f"Imbalance ratio:         {imbalance_ratio:.0f}x  (max/min tag count)")
print(f"Most common tag:         '{tag_counts_s.index[0]}'  ({int(max_count)} samples)")
print(f"Least common tag:        '{tag_counts_s.index[-1]}'  ({int(min_count)} samples)")
print(f"Tags with < 5 samples:   {(tag_counts_s < 5).sum()}")
print(f"Tags with < 20 samples:  {(tag_counts_s < 20).sum()}")
if has_cleanlab:
    print(f"CleanLab avg quality:    {avg_quality:.3f}")
    print(f"CleanLab pct flagged:    {pct_flagged*100:.1f}%")
    print(f"CleanLab min quality:    {min_quality:.3f}")

# ---- DYNAMIC THRESHOLDS -----------------------------------------------
# MIN_TAG_SAMPLES: scale with dataset size — tiny data needs lower bar
MIN_TAG_SAMPLES = max(3, min(10, int(n_total * 0.005)))

# KW_DROPOUT: higher imbalance = humans relied more on keywords = more dropout
KW_DROPOUT = min(0.5, 0.20 + (imbalance_ratio / 200))

# RARE_THRESHOLD: tag is "rare" if below 3% of dataset size
RARE_THRESHOLD = max(10, int(n_total * 0.03))

# DOMINANT_FREQ: tag appearing in > this fraction = dominant
DOMINANT_FREQ = 0.50

# RARE_FREQ: tag appearing in < this fraction = rare
RARE_FREQ = max(0.02, min(0.08, 1 / n_tags_raw))

# LABEL_SMOOTHING: driven by cleanlab quality
# Perfect quality (1.0) → 0.02, Very noisy (0.0) → 0.15
if has_cleanlab:
    BASE_SMOOTH = 0.02 + (1.0 - avg_quality) * 0.20
    BASE_SMOOTH = round(min(0.15, max(0.02, BASE_SMOOTH)), 3)
else:
    BASE_SMOOTH = 0.05

# GAMMA_NEG for ASL: scale with imbalance — more imbalanced = higher penalty
GAMMA_NEG = min(6, max(2, int(2 + imbalance_ratio / 30)))

# EPOCHS: more epochs for smaller datasets (less data = needs more passes)
EPOCHS = max(5, min(12, int(20000 / n_total + 3)))

# BATCH_SIZE: smaller dataset = smaller batches to get more gradient steps
BATCH_SIZE = 32 if n_total >= 5000 else 16 if n_total >= 1000 else 8

# MAX_LEN: set from actual summary length in data
if 'summary' in df.columns:
    median_chars = df['summary'].dropna().str.len().median()
    MAX_LEN = 512 if median_chars > 1200 else 384 if median_chars > 600 else 256
else:
    MAX_LEN = 256

# EARLY_STOPPING: more patience for small datasets
EARLY_STOP_PAT = 3 if n_total >= 2000 else 4

# UNDERSAMPLE: only if dominant tags are truly dominant
DO_UNDERSAMPLE = imbalance_ratio > 5 and (tag_counts_s / n_total > DOMINANT_FREQ).any()

# AUGMENT: only if there are rare tags to augment
DO_AUGMENT = (tag_counts_s < RARE_THRESHOLD).any()

LR = 2e-5

print(f"\n=== AUTO-COMPUTED SETTINGS ===")
print(f"MIN_TAG_SAMPLES:  {MIN_TAG_SAMPLES}  (tags below this dropped)")
print(f"RARE_THRESHOLD:   {RARE_THRESHOLD}  (tags below this get augmented)")
print(f"DOMINANT_FREQ:    {DOMINANT_FREQ:.0%}  (tags above this = dominant)")
print(f"RARE_FREQ:        {RARE_FREQ:.2%}  (tags below this = rare)")
print(f"KW_DROPOUT:       {KW_DROPOUT:.2f}  (higher imbalance → more keyword masking)")
print(f"BASE_SMOOTH:      {BASE_SMOOTH}  (label smoothing from cleanlab quality)")
print(f"GAMMA_NEG (ASL):  {GAMMA_NEG}  (higher imbalance → stronger penalty)")
print(f"EPOCHS:           {EPOCHS}")
print(f"BATCH_SIZE:       {BATCH_SIZE}")
print(f"MAX_LEN:          {MAX_LEN}  (set from median summary length)")
print(f"EARLY_STOP_PAT:   {EARLY_STOP_PAT}")
print(f"DO_AUGMENT:       {DO_AUGMENT}")
print(f"DO_UNDERSAMPLE:   {DO_UNDERSAMPLE}")

# ============================================================
# STEP 5 — DROP EXTREMELY RARE TAGS
# ============================================================
valid_mask = y.sum(axis=0) >= MIN_TAG_SAMPLES
dropped    = mlb.classes_[~valid_mask]
valid_tags = mlb.classes_[valid_mask]
y          = y[:, valid_mask]
NUM_LABELS = len(valid_tags)
print(f"\nKept {NUM_LABELS} tags, dropped {len(dropped)} with < {MIN_TAG_SAMPLES} samples")
if len(dropped): print(f"  Dropped: {list(dropped)}")

keep = y.sum(axis=1) > 0
df, y = df[keep].reset_index(drop=True), y[keep]
print(f"Complaints after tag filtering: {len(df)}")

# ============================================================
# STEP 6 — CLEANLAB NOISE SCORE PER ROW
# ============================================================
if has_cleanlab:
    for col in ['has_label_issue', 'cleanlab_has_issue']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.upper().map(
                {'TRUE': True, 'FALSE': False, '1': True, '0': False}
            ).fillna(False)
    # noise_score: 0=clean, 1=very noisy
    df['noise_score'] = (1.0 - df['cleanlab_avg_quality'].fillna(avg_quality).clip(0,1))
else:
    df['noise_score'] = 0.05
    print("No CleanLab columns found — using default noise_score=0.05")

# ============================================================
# STEP 7 — TRAIN / VAL / TEST SPLIT
# ============================================================
idx = np.arange(len(df))
tr_idx, temp   = train_test_split(idx, test_size=0.2, random_state=42)
val_idx, te_idx = train_test_split(temp, test_size=0.5, random_state=42)
df_tr,  y_tr  = df.iloc[tr_idx].reset_index(drop=True),  y[tr_idx]
df_val, y_val = df.iloc[val_idx].reset_index(drop=True), y[val_idx]
df_te,  y_te  = df.iloc[te_idx].reset_index(drop=True),  y[te_idx]
print(f"\nSplit → Train: {len(df_tr)} | Val: {len(df_val)} | Test: {len(df_te)}")

# ============================================================
# STEP 8 — AUGMENT RARE-TAG SAMPLES (train only, if needed)
# ============================================================
def augment_summary(text):
    words = str(text).split()
    if len(words) < 6: return text
    op = random.choice(['swap', 'drop', 'repeat'])
    if op == 'swap' and len(words) > 3:
        i = random.randint(0, len(words)-2)
        words[i], words[i+1] = words[i+1], words[i]
    elif op == 'drop' and len(words) > 8:
        words.pop(random.randint(1, len(words)-2))
    elif op == 'repeat':
        mid = len(words)//2
        words = words + words[mid:mid+6]
    return ' '.join(words)

if DO_AUGMENT:
    tag_counts_tr = y_tr.sum(axis=0)
    rare_idx      = np.where(tag_counts_tr < RARE_THRESHOLD)[0]
    aug_rows, aug_y = [], []
    for i in range(len(df_tr)):
        if y_tr[i, rare_idx].any():
            r = df_tr.iloc[i].copy()
            r['summary'] = augment_summary(str(r.get('summary', '')))
            aug_rows.append(r); aug_y.append(y_tr[i])
    if aug_rows:
        df_tr = pd.concat([df_tr, pd.DataFrame(aug_rows)], ignore_index=True)
        y_tr  = np.vstack([y_tr, np.array(aug_y)])
        print(f"Augmented {len(aug_rows)} rare-tag samples → train: {len(df_tr)}")
else:
    print("Augmentation skipped — no rare tags detected")

# ============================================================
# STEP 9 — UNDERSAMPLE DOMINANT-TAG-ONLY (if needed)
# ============================================================
if DO_UNDERSAMPLE:
    tag_freq_tr  = y_tr.sum(axis=0) / len(y_tr)
    dom_set      = set(np.where(tag_freq_tr > DOMINANT_FREQ)[0])
    keep_i = [i for i in range(len(df_tr))
               if not set(np.where(y_tr[i]==1)[0]).issubset(dom_set)
               or random.random() < 0.5]
    removed = len(df_tr) - len(keep_i)
    df_tr = df_tr.iloc[keep_i].reset_index(drop=True)
    y_tr  = y_tr[keep_i]
    print(f"Undersampled {removed} dominant-only rows → train: {len(df_tr)}")
else:
    print("Undersampling skipped — no dominant tags detected")

# ============================================================
# STEP 10 — WEIGHTED SAMPLER
# ============================================================
tag_w = 1.0 / (y_tr.sum(axis=0) + 1e-6)
if 'sample_weight' in df_tr.columns:
    sw = df_tr['sample_weight'].fillna(1.0).values.astype(np.float32)
    sw = sw * (y_tr @ tag_w + 1e-6)
    print(f"\nUsing existing sample_weight × rare-tag boost")
else:
    sw = (y_tr @ tag_w).astype(np.float32)
    print(f"\nComputed sample weights from inverse tag frequency")

sampler = WeightedRandomSampler(torch.tensor(sw, dtype=torch.float32), len(sw), replacement=True)

# ============================================================
# STEP 11 — BUILD INPUT TEXT (using your actual columns)
# summary = always (full complaint text, your richest signal)
# Service = always (categorical context)
# tier2_consolidated_description = always (semantic context for tags)
# keyword_group = with dynamic dropout (context only, NOT predicted)
# tier_1 = NOT used (collinear with keyword_group)
# clean_summary = NOT used (you want full summary)
# ============================================================
def build_text(row, is_training=False):
    parts = []
    svc = str(row.get('Service', '')).strip()
    if svc and svc.lower() not in ('nan', 'none', ''):
        parts.append(f"Service: {svc}")
    summ = str(row.get('summary', '')).strip()
    if summ and summ.lower() not in ('nan', 'none', ''):
        parts.append(f"Complaint: {summ}")
    desc = str(row.get('tier2_consolidated_description', '')).strip()
    if desc and desc.lower() not in ('nan', 'none', ''):
        parts.append(f"Context: {desc}")
    # keyword_group: context hint only, masked during training
    if not is_training or random.random() > KW_DROPOUT:
        kg = row.get('keyword_group_parsed', [])
        if isinstance(kg, list) and kg:
            parts.append(f"Categories: {', '.join(str(x) for x in kg)}")
    return " [SEP] ".join(parts)

# ============================================================
# STEP 12 — DATASET WITH ADAPTIVE LABEL SMOOTHING
# ============================================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

class ComplaintDataset(Dataset):
    def __init__(self, df_rows, labels, is_training=False):
        self.df, self.labels = df_rows.reset_index(drop=True), labels
        self.is_training = is_training

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row  = self.df.iloc[idx]
        text = build_text(row, is_training=self.is_training)
        enc  = tokenizer(text, max_length=MAX_LEN, padding='max_length',
                         truncation=True, return_tensors='pt')
        lbl  = torch.tensor(self.labels[idx], dtype=torch.float32)
        if self.is_training:
            # Adaptive per-row smoothing: clean rows barely smoothed, noisy rows more
            noise  = float(row.get('noise_score', BASE_SMOOTH))
            smooth = BASE_SMOOTH + noise * 0.10
            smooth = min(0.20, max(0.01, smooth))
            lbl    = lbl * (1 - smooth) + smooth * 0.5
        return {'input_ids': enc['input_ids'].squeeze(),
                'attention_mask': enc['attention_mask'].squeeze(),
                'labels': lbl}

train_ds = ComplaintDataset(df_tr,  y_tr,  is_training=True)
val_ds   = ComplaintDataset(df_val, y_val, is_training=False)
test_ds  = ComplaintDataset(df_te,  y_te,  is_training=False)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,  num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,    num_workers=0)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,    num_workers=0)
print(f"Loaders → Train: {len(train_loader)} | Val: {len(val_loader)} | Test: {len(test_loader)} batches")

# ============================================================
# STEP 13 — MODEL
# ============================================================
class ComplaintTagger(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert       = AutoModel.from_pretrained(MODEL_NAME)
        self.dropout    = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, NUM_LABELS)
    def forward(self, input_ids, attention_mask):
        out    = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(out.last_hidden_state[:, 0, :])
        return self.classifier(pooled)

model = ComplaintTagger().to(DEVICE)
print(f"\nModel params: {sum(p.numel() for p in model.parameters()):,} (66M = normal for DistilBERT)")

# ============================================================
# STEP 14 — ASYMMETRIC LOSS (gamma_neg auto-scaled from imbalance)
# ============================================================
class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05):
        super().__init__()
        self.gn, self.gp, self.clip = gamma_neg, gamma_pos, clip
    def forward(self, logits, targets):
        probs     = torch.sigmoid(logits)
        probs_neg = (1 - probs + self.clip).clamp(max=1)
        loss_pos  = targets       * torch.log(probs.clamp(1e-8))      * (1-probs)**self.gp
        loss_neg  = (1-targets)   * torch.log(probs_neg.clamp(1e-8)) * probs**self.gn
        return (-loss_pos - loss_neg).mean()

criterion = AsymmetricLoss(gamma_neg=GAMMA_NEG, gamma_pos=1, clip=0.05)
print(f"AsymmetricLoss: gamma_neg={GAMMA_NEG} (auto-scaled from {imbalance_ratio:.0f}x imbalance)")

# ============================================================
# STEP 15 — OPTIMIZER + SCHEDULER
# ============================================================
optimizer   = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
total_steps = len(train_loader) * EPOCHS
scheduler   = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=max(1, total_steps // 10),
    num_training_steps=total_steps
)

# ============================================================
# STEP 16 — TRAINING WITH EARLY STOPPING
# ============================================================
def train_epoch(model, loader):
    model.train(); total = 0
    for b in tqdm(loader, desc="  Train", leave=False):
        optimizer.zero_grad()
        loss = criterion(model(b['input_ids'].to(DEVICE), b['attention_mask'].to(DEVICE)),
                         b['labels'].to(DEVICE))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step(); scheduler.step()
        total += loss.item()
    return total / len(loader)

def eval_epoch(model, loader):
    model.eval(); probs_all, labels_all = [], []
    with torch.no_grad():
        for b in tqdm(loader, desc="  Eval", leave=False):
            p = torch.sigmoid(model(b['input_ids'].to(DEVICE),
                                    b['attention_mask'].to(DEVICE))).cpu().numpy()
            probs_all.append(p); labels_all.append(b['labels'].numpy())
    return np.vstack(probs_all), np.vstack(labels_all)

print(f"\n=== TRAINING ({EPOCHS} epochs, patience={EARLY_STOP_PAT}) ===")
best_f1, pat = 0.0, 0
history = []

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    tl          = train_epoch(model, train_loader)
    vp, vl      = eval_epoch(model, val_loader)
    vl_hard     = (vl >= 0.5).astype(int)
    macro = f1_score(vl_hard, (vp >= 0.5).astype(int), average='macro',  zero_division=0)
    micro = f1_score(vl_hard, (vp >= 0.5).astype(int), average='micro',  zero_division=0)
    history.append({'epoch': epoch+1, 'loss': round(tl,4),
                    'macro_f1': round(macro,4), 'micro_f1': round(micro,4)})
    print(f"  Loss: {tl:.4f}  |  Val Macro F1: {macro:.4f}  |  Val Micro F1: {micro:.4f}")
    if macro > best_f1:
        best_f1 = macro
        torch.save(model.state_dict(), 'best_complaint_tagger.pt')
        print(f"  ✅ Best model saved  (Macro F1: {best_f1:.4f})")
        pat = 0
    else:
        pat += 1
        if pat >= EARLY_STOP_PAT: print(f"  ⏹ Early stop after {epoch+1} epochs"); break

pd.DataFrame(history).to_csv('training_history.csv', index=False)
print(f"\nTraining history saved to training_history.csv")

# ============================================================
# STEP 17 — PER-TAG THRESHOLD TUNING
# ============================================================
print("\n=== THRESHOLD TUNING ===")
model.load_state_dict(torch.load('best_complaint_tagger.pt'))
vp, vl  = eval_epoch(model, val_loader)
vl_hard = (vl >= 0.5).astype(int)
tag_freq = y_tr.sum(axis=0) / len(y_tr)
thresholds = []

for i in range(NUM_LABELS):
    best_t, best_f = 0.5, 0.0
    for t in np.arange(0.10, 0.91, 0.05):
        preds = (vp[:, i] >= t).astype(int)
        f     = f1_score(vl_hard[:, i], preds, zero_division=0)
        if f > best_f: best_f, best_t = f, t
    if tag_freq[i] > DOMINANT_FREQ: best_t = max(best_t, 0.65)
    if tag_freq[i] < RARE_FREQ:     best_t = min(best_t, 0.35)
    thresholds.append(best_t)

thresholds = np.array(thresholds)
thr_df = pd.DataFrame({'tag': valid_tags, 'frequency': tag_freq,
                        'threshold': thresholds, 'val_support': vl_hard.sum(axis=0)}
                       ).sort_values('frequency', ascending=False)
print(thr_df.to_string(index=False))

# ============================================================
# STEP 18 — FINAL TEST EVALUATION
# ============================================================
print("\n=== FINAL TEST RESULTS ===")
tp, tl  = eval_epoch(model, test_loader)
tl_hard = (tl >= 0.5).astype(int)
preds   = (tp >= thresholds).astype(int)

print(f"Macro F1:    {f1_score(tl_hard, preds, average='macro',   zero_division=0):.4f}  ← main metric")
print(f"Micro F1:    {f1_score(tl_hard, preds, average='micro',   zero_division=0):.4f}")
print(f"Samples F1:  {f1_score(tl_hard, preds, average='samples', zero_division=0):.4f}")
print()
print(classification_report(tl_hard, preds, target_names=valid_tags, zero_division=0))

# Save everything
np.save('tag_thresholds.npy', thresholds)
thr_df.to_csv('tag_info.csv', index=False)

pred_df = df_te[['case_reference']].copy()
for i, tag in enumerate(valid_tags):
    pred_df[f'pred_{tag}'] = preds[:, i]
    pred_df[f'prob_{tag}'] = tp[:, i].round(3)
pred_df.to_csv('test_predictions.csv', index=False)
print("\nSaved: best_complaint_tagger.pt | tag_thresholds.npy | tag_info.csv | test_predictions.csv | training_history.csv")

# ============================================================
# STEP 19 — INFERENCE
# ============================================================
def predict_tags(row_dict, top_k=None):
    row  = pd.Series(row_dict)
    text = build_text(row, is_training=False)
    enc  = tokenizer(text, max_length=MAX_LEN, padding='max_length',
                     truncation=True, return_tensors='pt')
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(
            model(enc['input_ids'].to(DEVICE), enc['attention_mask'].to(DEVICE))
        ).cpu().numpy()[0]
    pred_idx = np.where(probs >= thresholds)[0]
    if top_k and len(pred_idx) == 0:
        pred_idx = np.argsort(probs)[-top_k:]
    tags = sorted([valid_tags[i] for i in pred_idx])
    conf = {valid_tags[i]: round(float(probs[i]), 3)
            for i in sorted(pred_idx, key=lambda x: -probs[x])}
    return tags, conf
