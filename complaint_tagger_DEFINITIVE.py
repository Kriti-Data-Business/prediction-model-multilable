
# ============================================================
# DYNAMIC COMPLAINT TAGGER — DEFINITIVE PIPELINE v4
# Handles: severe imbalance, noisy labels, rare tags,
#          CleanLab (fast concatenated column method),
#          human review export, checkpoint saving,
#          embedding caching, full feedback loop
# ============================================================

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import ast, random, os, json
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score, classification_report
from tqdm import tqdm

DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_NAME = 'distilbert-base-uncased'
DATA_FILE  = 'cfpb_complaints_simulated.csv'

# Output folders
Path('checkpoints').mkdir(exist_ok=True)
Path('saved_data').mkdir(exist_ok=True)
Path('human_review').mkdir(exist_ok=True)

print(f"Device: {DEVICE} | Model: {MODEL_NAME}")
print("=" * 65)
print("DEFINITIVE DYNAMIC PIPELINE — checkpoints + CleanLab + human review")
print("=" * 65)

# ============================================================
# HELPER — resume from checkpoint if exists
# ============================================================
def checkpoint_exists(name):
    return Path(f'checkpoints/{name}').exists()

def save_checkpoint(obj, name):
    if isinstance(obj, pd.DataFrame):
        obj.to_csv(f'checkpoints/{name}', index=False)
    elif isinstance(obj, np.ndarray):
        np.save(f'checkpoints/{name}', obj)
    elif isinstance(obj, dict):
        with open(f'checkpoints/{name}', 'w') as f: json.dump(obj, f)
    print(f"  ✅ Saved checkpoint: {name}")

def load_checkpoint(name):
    p = f'checkpoints/{name}'
    if name.endswith('.csv'):      return pd.read_csv(p)
    if name.endswith('.npy'):      return np.load(p, allow_pickle=True)
    if name.endswith('.json'):
        with open(p) as f: return json.load(f)

# ============================================================
# STEP 1 — LOAD + PARSE
# ============================================================
print("\n[STEP 1] Loading and parsing data...")
df = pd.read_csv(DATA_FILE)

def parse_tags(val):
    if isinstance(val, list): return [t.strip() for t in val if str(t).strip()]
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
print(f"  Loaded: {len(df)} rows, {df.shape[1]} columns")

# ============================================================
# STEP 2 — BINARIZE LABELS
# ============================================================
print("\n[STEP 2] Binarizing tier_2 labels...")
mlb = MultiLabelBinarizer()
y   = mlb.fit_transform(df['tags_parsed'])
tag_counts_s    = pd.Series(y.sum(axis=0), index=mlb.classes_).sort_values(ascending=False)
n_total         = len(df)
imbalance_ratio = tag_counts_s.max() / max(tag_counts_s.min(), 1)
tags_per_row    = y.sum(axis=1)

# ============================================================
# STEP 3 — AUTO-DIAGNOSE → COMPUTE ALL SETTINGS
# ============================================================
print("\n[STEP 3] Diagnosing data quality and imbalance...")

HAS_CLEANLAB = 'cleanlab_avg_quality' in df.columns
if HAS_CLEANLAB:
    for col in ['has_label_issue', 'cleanlab_has_issue']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.upper()                             .map({'TRUE':True,'FALSE':False,'1':True,'0':False}).fillna(False)
    avg_quality = df['cleanlab_avg_quality'].fillna(0.9).mean()
    pct_flagged = df['has_label_issue'].mean() if 'has_label_issue' in df.columns else 0
    df['noise_score'] = (1.0 - df['cleanlab_avg_quality'].fillna(avg_quality).clip(0,1))
else:
    avg_quality, pct_flagged = 0.9, 0.0
    tc = y.sum(axis=1); exp = tc.mean(); std = tc.std() + 1e-6
    df['noise_score'] = (np.abs(tc - exp) / (std * 3)).clip(0, 0.5)

MIN_TAG_SAMPLES = max(3, min(10, int(n_total * 0.005)))
RARE_THRESHOLD  = max(10, int(n_total * 0.03))
DOMINANT_FREQ   = 0.50
RARE_FREQ       = max(0.02, min(0.08, 1.5 / len(mlb.classes_)))
KW_DROPOUT      = min(0.55, 0.20 + (min(imbalance_ratio, 100) / 200))
BASE_SMOOTH     = round(min(0.15, max(0.02, 0.02 + (1.0 - avg_quality) * 0.25)), 3)
GAMMA_NEG       = int(min(6, max(2, 2 + min(imbalance_ratio, 100) / 30)))
EPOCHS          = max(5, min(12, int(20000 / n_total + 3)))
BATCH_SIZE      = 32 if n_total >= 5000 else 16 if n_total >= 1000 else 8
EARLY_STOP_PAT  = 3 if n_total >= 2000 else 4
LR              = 2e-5
med_chars       = df['summary'].dropna().str.len().median() if 'summary' in df.columns else 600
MAX_LEN         = 512 if med_chars > 1200 else 384 if med_chars > 600 else 256
DO_AUGMENT      = (tag_counts_s < RARE_THRESHOLD).any()
DO_UNDERSAMPLE  = imbalance_ratio > 5 and (tag_counts_s / n_total > DOMINANT_FREQ).any()

# Print full diagnosis
print(f"""
{'='*65}
  DATA DIAGNOSIS REPORT
{'='*65}
  Total complaints:       {n_total}
  Unique tier_2 tags:     {len(mlb.classes_)}
  Avg tags/complaint:     {tags_per_row.mean():.2f}
  Imbalance ratio:        {imbalance_ratio:.0f}x
  Most common tag:        '{tag_counts_s.index[0]}' ({int(tag_counts_s.max())} samples)
  Least common tag:       '{tag_counts_s.index[-1]}' ({int(tag_counts_s.min())} samples)
  Tags < 5 samples:       {(tag_counts_s < 5).sum()}
  Tags < {RARE_THRESHOLD} samples:      {(tag_counts_s < RARE_THRESHOLD).sum()} → will be augmented
  CleanLab available:     {HAS_CLEANLAB}
  Avg label quality:      {avg_quality:.3f}
  Pct flagged:            {pct_flagged*100:.1f}%

  AUTO SETTINGS:
  MIN_TAG_SAMPLES={MIN_TAG_SAMPLES}  RARE_THRESHOLD={RARE_THRESHOLD}  KW_DROPOUT={KW_DROPOUT:.2f}
  BASE_SMOOTH={BASE_SMOOTH}  GAMMA_NEG={GAMMA_NEG}  EPOCHS={EPOCHS}
  BATCH_SIZE={BATCH_SIZE}  MAX_LEN={MAX_LEN}  LR={LR}
  DO_AUGMENT={DO_AUGMENT}  DO_UNDERSAMPLE={DO_UNDERSAMPLE}
{'='*65}""")

# ============================================================
# STEP 4 — DROP RARE TAGS
# ============================================================
print("[STEP 4] Filtering rare tags...")
valid_mask = y.sum(axis=0) >= MIN_TAG_SAMPLES
dropped    = mlb.classes_[~valid_mask]
valid_tags = mlb.classes_[valid_mask]
y          = y[:, valid_mask]
NUM_LABELS = len(valid_tags)
print(f"  Kept {NUM_LABELS} tags, dropped {len(dropped)}: {list(dropped) if len(dropped) else 'none'}")
keep = y.sum(axis=1) > 0
df, y = df[keep].reset_index(drop=True), y[keep]
print(f"  Complaints after filter: {len(df)}")

# ============================================================
# STEP 5 — CLEANLAB (FAST — SINGLE CONCATENATED COLUMN)
# Runs only if cleanlab_avg_quality not already in data
# Saves results so you never need to rerun
# ============================================================
CLEANLAB_CACHE = 'checkpoints/cleanlab_scores.csv'

if HAS_CLEANLAB:
    print("\n[STEP 5] CleanLab scores already in data — skipping computation")
elif checkpoint_exists('cleanlab_scores.csv'):
    print("\n[STEP 5] Loading cached CleanLab scores...")
    cl_scores = load_checkpoint('cleanlab_scores.csv')
    df['cleanlab_avg_quality'] = cl_scores['cleanlab_avg_quality'].values
    df['cleanlab_has_issue']   = cl_scores['cleanlab_has_issue'].values
    df['noise_score']          = (1.0 - df['cleanlab_avg_quality'].clip(0,1))
    print(f"  Loaded scores for {len(cl_scores)} rows")
else:
    print("\n[STEP 5] Running CleanLab (fast concatenated-column method)...")

    # ── SMART WAY: one concatenated column → TF-IDF → CleanLab ──
    df['cleanlab_input'] = (
        df['Service'].fillna('') + ' | ' +
        df['keywords_list'].astype(str).fillna('') + ' | ' +
        df['summary'].fillna('')   # summary last = richest signal
    )

    # Cap max_features=15000 → cuts time in half vs default 50k+
    tfidf = TfidfVectorizer(max_features=15000, ngram_range=(1,2), sublinear_tf=True)
    X_cl  = tfidf.fit_transform(df['cleanlab_input'])
    print(f"  TF-IDF matrix: {X_cl.shape}  (fast, capped at 15k features)")

    try:
        from cleanlab.multilabel_classification.label_quality_scores import (
            multilabel_label_quality_scores
        )
        # Cross-val predicted probabilities (required by CleanLab)
        clf_cl = OneVsRestClassifier(
            LogisticRegression(class_weight='balanced', max_iter=300, C=1.0),
            n_jobs=-1
        )
        print("  Computing cross-val probabilities (5-fold, ~2 mins)...")
        pred_probs = cross_val_predict(clf_cl, X_cl, y, cv=5, method='predict_proba')
        quality    = multilabel_label_quality_scores(y, pred_probs)
        df['cleanlab_avg_quality'] = quality
        df['cleanlab_has_issue']   = quality < 0.7
        df['noise_score']          = (1.0 - quality.clip(0,1))

        cl_out = df[['case_reference','cleanlab_avg_quality','cleanlab_has_issue','noise_score']]
        cl_out.to_csv(CLEANLAB_CACHE, index=False)

        print(f"  Mean quality: {quality.mean():.3f}")
        print(f"  Flagged noisy (quality<0.7): {(quality<0.7).sum()} ({(quality<0.7).mean()*100:.1f}%)")
        print(f"  Very noisy (quality<0.4):    {(quality<0.4).sum()}")
        save_checkpoint(cl_out, 'cleanlab_scores.csv')

    except ImportError:
        print("  cleanlab not installed — pip install cleanlab")
        print("  Using tag-count consistency as noise proxy instead")
        tc = y.sum(axis=1); exp = tc.mean(); std = tc.std() + 1e-6
        df['noise_score'] = (np.abs(tc - exp) / (std * 3)).clip(0, 0.5)

# ── Save cleaned dataframe after CleanLab step ──────────────
df.to_csv('saved_data/df_after_cleanlab.csv', index=False)
np.save('saved_data/y_labels.npy', y)
with open('saved_data/valid_tags.json', 'w') as f: json.dump(list(valid_tags), f)
print("  💾 Saved: saved_data/df_after_cleanlab.csv | y_labels.npy | valid_tags.json")

# ============================================================
# STEP 6 — TRAIN/VAL/TEST SPLIT
# ============================================================
print("\n[STEP 6] Splitting data...")
idx = np.arange(len(df))
tr_idx, temp    = train_test_split(idx, test_size=0.2, random_state=42)
val_idx, te_idx = train_test_split(temp, test_size=0.5, random_state=42)
df_tr,  y_tr  = df.iloc[tr_idx].reset_index(drop=True), y[tr_idx]
df_val, y_val = df.iloc[val_idx].reset_index(drop=True), y[val_idx]
df_te,  y_te  = df.iloc[te_idx].reset_index(drop=True), y[te_idx]
print(f"  Train: {len(df_tr)} | Val: {len(df_val)} | Test: {len(df_te)}")

# Save splits for reproducibility
df_tr.to_csv('saved_data/train_split.csv', index=False)
df_val.to_csv('saved_data/val_split.csv',  index=False)
df_te.to_csv('saved_data/test_split.csv',  index=False)
np.save('saved_data/y_tr.npy', y_tr)
np.save('saved_data/y_val.npy', y_val)
np.save('saved_data/y_te.npy', y_te)
print("  💾 Saved: train/val/test splits + label arrays")

# ============================================================
# STEP 7 — IMBALANCE: TEXT AUGMENTATION (rare tags, train only)
# ============================================================
print("\n[STEP 7] Handling imbalance — augmentation...")
def augment_text(text):
    words = str(text).split()
    if len(words) < 6: return text
    op = random.choice(['swap', 'drop', 'repeat'])
    if op == 'swap' and len(words) > 3:
        i = random.randint(0, len(words)-2)
        words[i], words[i+1] = words[i+1], words[i]
    elif op == 'drop' and len(words) > 8:
        words.pop(random.randint(1, len(words)-2))
    elif op == 'repeat':
        mid = len(words)//2; words = words + words[mid:mid+6]
    return ' '.join(words)

if DO_AUGMENT:
    tag_counts_tr = y_tr.sum(axis=0)
    rare_idx      = np.where(tag_counts_tr < RARE_THRESHOLD)[0]
    aug_rows, aug_y = [], []
    for i in range(len(df_tr)):
        if y_tr[i, rare_idx].any():
            r = df_tr.iloc[i].copy()
            r['summary'] = augment_text(str(r.get('summary', '')))
            aug_rows.append(r); aug_y.append(y_tr[i])
    if aug_rows:
        df_tr = pd.concat([df_tr, pd.DataFrame(aug_rows)], ignore_index=True)
        y_tr  = np.vstack([y_tr, np.array(aug_y)])
        print(f"  Augmented {len(aug_rows)} rare-tag rows → train: {len(df_tr)}")
else:
    print("  No augmentation needed")

# ============================================================
# STEP 8 — IMBALANCE: UNDERSAMPLE DOMINANT-ONLY ROWS
# ============================================================
print("\n[STEP 8] Handling imbalance — undersampling...")
if DO_UNDERSAMPLE:
    tag_freq_tr = y_tr.sum(axis=0) / len(y_tr)
    dom_set     = set(np.where(tag_freq_tr > DOMINANT_FREQ)[0])
    keep_i      = [i for i in range(len(df_tr))
                   if not set(np.where(y_tr[i]==1)[0]).issubset(dom_set)
                   or random.random() < 0.5]
    removed = len(df_tr) - len(keep_i)
    df_tr = df_tr.iloc[keep_i].reset_index(drop=True)
    y_tr  = y_tr[keep_i]
    print(f"  Removed {removed} dominant-only rows → train: {len(df_tr)}")
else:
    print("  Undersampling skipped — no dominant tags")

# ============================================================
# STEP 9 — IMBALANCE: WEIGHTED SAMPLER
# ============================================================
print("\n[STEP 9] Setting up weighted sampler...")
tag_w = 1.0 / (y_tr.sum(axis=0) + 1e-6)
if 'sample_weight' in df_tr.columns:
    sw = df_tr['sample_weight'].fillna(1.0).values.astype(np.float32)
    sw = sw * (y_tr @ tag_w + 1e-6)
    print(f"  Using existing sample_weight × rare-tag boost")
else:
    sw = (y_tr @ tag_w).astype(np.float32)
    print(f"  Computed from inverse tag frequency")
sampler = WeightedRandomSampler(torch.tensor(sw, dtype=torch.float32), len(sw), replacement=True)

# ============================================================
# STEP 10 — BUILD INPUT TEXT
# summary       → always (full complaint, richest signal)
# Service       → always
# tier2_consolidated_description → always
# keyword_group → context only, with KW_DROPOUT masking
# NOT used: clean_summary, tier_1, case_reference, referral_date
# ============================================================
def build_text(row, is_training=False):
    parts = []
    for label, col in [('Service', 'Service'), ('Complaint', 'summary'),
                        ('Context', 'tier2_consolidated_description')]:
        val = str(row.get(col, '')).strip()
        if val and val.lower() not in ('nan', 'none', ''):
            parts.append(f"{label}: {val}")
    if not is_training or random.random() > KW_DROPOUT:
        kg = row.get('keyword_group_parsed', [])
        if isinstance(kg, list) and kg:
            parts.append(f"Categories: {', '.join(str(x) for x in kg)}")
    return ' [SEP] '.join(parts)

# ============================================================
# STEP 11 — DATASET WITH ADAPTIVE LABEL SMOOTHING
# ============================================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

class ComplaintDataset(Dataset):
    def __init__(self, df_rows, labels, is_training=False):
        self.df = df_rows.reset_index(drop=True)
        self.labels = labels; self.is_training = is_training
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        enc = tokenizer(build_text(row, self.is_training),
                        max_length=MAX_LEN, padding='max_length',
                        truncation=True, return_tensors='pt')
        lbl = torch.tensor(self.labels[idx], dtype=torch.float32)
        if self.is_training:
            ns     = float(row.get('noise_score', BASE_SMOOTH))
            smooth = min(0.20, max(0.01, BASE_SMOOTH + ns * 0.10))
            lbl    = lbl * (1-smooth) + smooth * 0.5
        return {'input_ids': enc['input_ids'].squeeze(),
                'attention_mask': enc['attention_mask'].squeeze(), 'labels': lbl}

train_loader = DataLoader(ComplaintDataset(df_tr, y_tr, True),
                          batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
val_loader   = DataLoader(ComplaintDataset(df_val, y_val), batch_size=BATCH_SIZE, num_workers=0)
test_loader  = DataLoader(ComplaintDataset(df_te, y_te),  batch_size=BATCH_SIZE, num_workers=0)
print(f"\n[STEP 11] Loaders → Train:{len(train_loader)} Val:{len(val_loader)} Test:{len(test_loader)}")

# ============================================================
# STEP 12 — MODEL
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
print(f"\n[STEP 12] Model: {sum(p.numel() for p in model.parameters()):,} params (66M = normal DistilBERT)")

# ============================================================
# STEP 13 — ASYMMETRIC LOSS
# ============================================================
class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05):
        super().__init__()
        self.gn, self.gp, self.clip = gamma_neg, gamma_pos, clip
    def forward(self, logits, targets):
        p  = torch.sigmoid(logits)
        pn = (1-p+self.clip).clamp(max=1)
        return (-(targets * torch.log(p.clamp(1e-8)) * (1-p)**self.gp
                + (1-targets) * torch.log(pn.clamp(1e-8)) * p**self.gn)).mean()

criterion = AsymmetricLoss(gamma_neg=GAMMA_NEG)
optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer,
    num_warmup_steps=max(1, total_steps//10), num_training_steps=total_steps)

# ============================================================
# STEP 14 — TRAINING WITH EPOCH CHECKPOINTS
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
            probs_all.append(
                torch.sigmoid(model(b['input_ids'].to(DEVICE),
                                    b['attention_mask'].to(DEVICE))).cpu().numpy())
            labels_all.append(b['labels'].numpy())
    return np.vstack(probs_all), np.vstack(labels_all)

print(f"\n[STEP 14] Training ({EPOCHS} epochs, patience={EARLY_STOP_PAT})...")
best_f1, pat = 0.0, 0; history = []

for epoch in range(EPOCHS):
    print(f"\n  Epoch {epoch+1}/{EPOCHS}")
    tl     = train_epoch(model, train_loader)
    vp, vl = eval_epoch(model, val_loader)
    vl_h   = (vl >= 0.5).astype(int)
    macro  = f1_score(vl_h, (vp >= 0.5).astype(int), average='macro',  zero_division=0)
    micro  = f1_score(vl_h, (vp >= 0.5).astype(int), average='micro',  zero_division=0)
    history.append({'epoch': epoch+1, 'loss': round(tl,4),
                    'macro_f1': round(macro,4), 'micro_f1': round(micro,4)})
    print(f"  Loss:{tl:.4f}  Val Macro F1:{macro:.4f}  Micro F1:{micro:.4f}")

    # Save every epoch checkpoint
    torch.save({'epoch': epoch+1, 'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(), 'macro_f1': macro},
               f'checkpoints/epoch_{epoch+1}.pt')

    if macro > best_f1:
        best_f1 = macro
        torch.save(model.state_dict(), 'checkpoints/best_model.pt')
        print(f"  ✅ Best model saved (Macro F1: {best_f1:.4f})")
        pat = 0
    else:
        pat += 1
        if pat >= EARLY_STOP_PAT:
            print(f"  ⏹ Early stop — best Macro F1: {best_f1:.4f}"); break

pd.DataFrame(history).to_csv('saved_data/training_history.csv', index=False)
print("  💾 Saved: epoch checkpoints | best_model.pt | training_history.csv")

# ============================================================
# STEP 15 — PER-TAG THRESHOLD TUNING
# ============================================================
print("\n[STEP 15] Tuning per-tag thresholds on validation set...")
model.load_state_dict(torch.load('checkpoints/best_model.pt'))
vp, vl   = eval_epoch(model, val_loader)
vl_h     = (vl >= 0.5).astype(int)
tag_freq  = y_tr.sum(axis=0) / len(y_tr)
thresholds = []

for i in range(NUM_LABELS):
    best_t, best_f = 0.5, 0.0
    for t in np.arange(0.10, 0.91, 0.05):
        f = f1_score(vl_h[:,i], (vp[:,i]>=t).astype(int), zero_division=0)
        if f > best_f: best_f, best_t = f, t
    if tag_freq[i] > DOMINANT_FREQ: best_t = max(best_t, 0.65)
    if tag_freq[i] < RARE_FREQ:     best_t = min(best_t, 0.35)
    thresholds.append(best_t)

thresholds = np.array(thresholds)
thr_df = pd.DataFrame({
    'tag': valid_tags,
    'train_freq_%': (tag_freq*100).round(1),
    'threshold': thresholds,
    'category': ['dominant' if f>DOMINANT_FREQ else 'rare' if f<RARE_FREQ else 'normal'
                  for f in tag_freq]
}).sort_values('train_freq_%', ascending=False)

np.save('saved_data/tag_thresholds.npy', thresholds)
thr_df.to_csv('saved_data/tag_info.csv', index=False)
print("  Threshold summary:")
print(thr_df.to_string(index=False))

# ============================================================
# STEP 16 — FINAL TEST EVALUATION
# ============================================================
print("\n[STEP 16] Final test evaluation...")
tp, tl  = eval_epoch(model, test_loader)
tl_h    = (tl >= 0.5).astype(int)
preds   = (tp >= thresholds).astype(int)
macro_test = f1_score(tl_h, preds, average='macro',   zero_division=0)
micro_test = f1_score(tl_h, preds, average='micro',   zero_division=0)
samp_test  = f1_score(tl_h, preds, average='samples', zero_division=0)
print(f"  Macro F1:   {macro_test:.4f}  ← main metric")
print(f"  Micro F1:   {micro_test:.4f}")
print(f"  Samples F1: {samp_test:.4f}")
print()
print(classification_report(tl_h, preds, target_names=valid_tags, zero_division=0))

# Save test predictions
pred_df = df_te[['case_reference']].copy()
for i, tag in enumerate(valid_tags):
    pred_df[f'pred_{tag}'] = preds[:, i]
    pred_df[f'prob_{tag}'] = tp[:, i].round(3)
pred_df.to_csv('saved_data/test_predictions.csv', index=False)

# ============================================================
# STEP 17 — HUMAN REVIEW EXPORT
# Exports 3 categories of complaints for manual feedback:
# A) Most uncertain predictions (model needs human help)
# B) Rare-tag predictions (verify model is right)
# C) High-confidence wrong predictions (model overconfident)
# ============================================================
print("\n[STEP 17] Generating human review export...")

# A) Uncertainty score per complaint = avg entropy across tags
def prediction_entropy(probs):
    p = np.clip(probs, 1e-9, 1-1e-9)
    return -(p * np.log(p) + (1-p) * np.log(1-p)).mean(axis=1)

entropy = prediction_entropy(tp)
n_review = min(100, max(20, int(len(df_te) * 0.20)))

# A) Most uncertain
uncertain_idx = np.argsort(entropy)[-n_review//3:]
df_uncertain  = df_te.iloc[uncertain_idx][['case_reference','summary','Service','tags_parsed']].copy()
df_uncertain['model_predicted_tags'] = [
    ', '.join(sorted([valid_tags[j] for j in np.where(preds[i]==1)[0]]))
    for i in uncertain_idx]
df_uncertain['model_confidence']     = entropy[uncertain_idx].round(3)
df_uncertain['review_reason']        = 'Uncertain prediction — low confidence'
df_uncertain['human_corrected_tags'] = ''   # ← human fills this in

# B) Rare-tag predictions
rare_tag_idx_set = set(np.where(tag_freq < RARE_FREQ)[0])
rare_pred_rows   = [i for i in range(len(df_te))
                    if any(preds[i,j]==1 for j in rare_tag_idx_set)][:n_review//3]
df_rare          = df_te.iloc[rare_pred_rows][['case_reference','summary','Service','tags_parsed']].copy()
df_rare['model_predicted_tags'] = [
    ', '.join(sorted([valid_tags[j] for j in np.where(preds[i]==1)[0]]))
    for i in rare_pred_rows]
df_rare['model_confidence']     = [round(float(tp[i].max()), 3) for i in rare_pred_rows]
df_rare['review_reason']        = 'Rare tag predicted — please verify'
df_rare['human_corrected_tags'] = ''

# C) High-confidence (top probabilities — spot-check quality)
high_conf_idx = np.argsort(-tp.max(axis=1))[:n_review//3]
df_highconf   = df_te.iloc[high_conf_idx][['case_reference','summary','Service','tags_parsed']].copy()
df_highconf['model_predicted_tags'] = [
    ', '.join(sorted([valid_tags[j] for j in np.where(preds[i]==1)[0]]))
    for i in high_conf_idx]
df_highconf['model_confidence']     = tp[high_conf_idx].max(axis=1).round(3)
df_highconf['review_reason']        = 'High confidence — spot-check for overconfidence'
df_highconf['human_corrected_tags'] = ''

# Combine and export
review_df = pd.concat([df_uncertain, df_rare, df_highconf], ignore_index=True)
review_df = review_df.drop_duplicates(subset=['case_reference'])
review_path = 'human_review/complaints_for_review.csv'
review_df.to_csv(review_path, index=False)

print(f"""
  Human Review File: {review_path}
  Total rows exported: {len(review_df)}
    - Uncertain predictions: {len(df_uncertain)}
    - Rare-tag predictions:  {len(df_rare)}
    - High-confidence:       {len(df_highconf)}

  HOW TO USE:
  1. Open human_review/complaints_for_review.csv
  2. Read 'summary' + 'model_predicted_tags'
  3. Fill 'human_corrected_tags' column with correct tier_2 tags
  4. Run the FEEDBACK LOOP below to retrain on corrections
""")

# ============================================================
# STEP 18 — FEEDBACK LOOP (run after human review is complete)
# Load corrected CSV, merge back into training data, retrain
# ============================================================
def apply_human_feedback(corrected_csv_path):
    """
    Run this AFTER filling in human_review/complaints_for_review.csv
    It merges corrections back into training data and triggers retraining.
    Call: apply_human_feedback('human_review/complaints_for_review.csv')
    """
    corrected = pd.read_csv(corrected_csv_path)
    corrected = corrected[corrected['human_corrected_tags'].notna() &
                          (corrected['human_corrected_tags'].str.strip() != '')]

    if corrected.empty:
        print("No corrections found — fill the 'human_corrected_tags' column first"); return

    print(f"Applying {len(corrected)} human corrections...")
    corrected['tags_parsed'] = corrected['human_corrected_tags'].apply(
        lambda x: [t.strip() for t in str(x).split(',') if t.strip()])

    # Merge corrections into training data
    df_feedback = pd.read_csv('saved_data/train_split.csv')
    df_feedback['tags_parsed'] = df_feedback['tags_parsed'].apply(parse_tags)

    for _, row in corrected.iterrows():
        match = df_feedback['case_reference'] == row['case_reference']
        if match.any():
            df_feedback.loc[match, 'tags_parsed'] = str(row['tags_parsed'])
        else:
            # New annotated row — add to training set
            df_feedback = pd.concat([df_feedback, pd.DataFrame([row])], ignore_index=True)

    df_feedback.to_csv('saved_data/train_with_feedback.csv', index=False)
    print(f"Saved updated training set: saved_data/train_with_feedback.csv")
    print("Re-run pipeline with DATA_FILE = 'saved_data/train_with_feedback.csv' to retrain")

print("\n  Feedback function ready: call apply_human_feedback('human_review/complaints_for_review.csv')")
print("  after completing your human review.")

# ============================================================
# STEP 19 — INFERENCE ON NEW COMPLAINT
# ============================================================
def predict_tags(row_dict, top_k=None):
    """
    row_dict: {summary, Service, tier2_consolidated_description, keyword_group_parsed}
    Returns: (list of predicted tags, confidence dict)
    """
    row  = pd.Series(row_dict)
    text = build_text(row, is_training=False)
    enc  = tokenizer(text, max_length=MAX_LEN, padding='max_length',
                     truncation=True, return_tensors='pt')
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(enc['input_ids'].to(DEVICE),
                                    enc['attention_mask'].to(DEVICE))).cpu().numpy()[0]
    idx  = np.where(probs >= thresholds)[0]
    if top_k and len(idx) == 0:
        idx = np.argsort(probs)[-top_k:]
    tags = sorted([valid_tags[i] for i in idx])
    conf = {valid_tags[i]: round(float(probs[i]),3)
            for i in sorted(idx, key=lambda x: -probs[x])}
    return tags, conf

print("\n" + "="*65)
print("PIPELINE COMPLETE — all outputs saved to:")
print("  checkpoints/   → model checkpoints (every epoch + best)")
print("  saved_data/    → splits, labels, tags, thresholds, predictions")
print("  human_review/  → CSV for manual correction + feedback loop")
print("="*65)
