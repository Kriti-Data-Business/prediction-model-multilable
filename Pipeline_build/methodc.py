import pandas as pd
import numpy as np
import ast
from sentence_transformers import SentenceTransformer, util
from collections import Counter

# ─────────────────────────────────────────────────────────
# SETUP — Load model once (downloads ~80MB on first run)
# ─────────────────────────────────────────────────────────
print("Loading embedding model...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')  # fast, local, no API needed

# ─────────────────────────────────────────────────────────
# LOAD YOUR DATA
# ─────────────────────────────────────────────────────────
df = pd.read_csv("your_data.csv")
df['keywords_list'] = df['keywords_list'].apply(ast.literal_eval)

# ─────────────────────────────────────────────────────────
# LOAD YOUR KEYWORD MAPPING FILE
# Expected columns: canonical_keyword | description | example_complaints
# example_complaints: multiple examples separated by |
# e.g. "I found charges I didn't make | Someone used my account without permission"
# ─────────────────────────────────────────────────────────
mapping_df = pd.read_csv("keyword_mapping.csv")

MASTER_KEYWORDS  = mapping_df['canonical_keyword'].tolist()
KEYWORD_DESC     = dict(zip(mapping_df['canonical_keyword'], mapping_df['description']))
KEYWORD_EXAMPLES = dict(zip(mapping_df['canonical_keyword'], mapping_df['example_complaints']))

# ─────────────────────────────────────────────────────────
# PRE-ENCODE ALL KEYWORDS (do this ONCE before the main loop)
# Each keyword gets 2 embeddings:
#   1. Description embedding  — what the keyword means
#   2. Example embedding      — what complaints about this keyword sound like
# ─────────────────────────────────────────────────────────
print("Pre-encoding keyword descriptions and examples...")

kw_desc_embeddings    = {}
kw_example_embeddings = {}

for kw in MASTER_KEYWORDS:
    # 1. Description embedding
    desc_text = f"{kw}: {KEYWORD_DESC.get(kw, kw)}"
    kw_desc_embeddings[kw] = embedder.encode(desc_text, convert_to_tensor=True)

    # 2. Example complaints embedding (averaged across all examples)
    examples_raw = str(KEYWORD_EXAMPLES.get(kw, ""))
    example_list = [e.strip() for e in examples_raw.split('|') if e.strip()]

    if example_list:
        example_embs = embedder.encode(example_list, convert_to_tensor=True)
        kw_example_embeddings[kw] = example_embs.mean(dim=0)   # average = "typical" complaint
    else:
        kw_example_embeddings[kw] = kw_desc_embeddings[kw]     # fallback to description

print(f"Encoded {len(MASTER_KEYWORDS)} keywords ✅")

# ─────────────────────────────────────────────────────────
# CORE FUNCTION — Validate one keyword against one complaint
# ─────────────────────────────────────────────────────────
DESC_THRESHOLD    = 0.40   # similarity to keyword description
EXAMPLE_THRESHOLD = 0.38   # similarity to example complaints

def validate_keyword(complaint_text, keyword):
    """
    Returns verdict + reason for one keyword on one complaint.

    How it works:
      - Encodes the complaint into a vector
      - Compares it to the keyword description vector  → desc_sim
      - Compares it to the example complaints vector  → example_sim
      - Uses BOTH scores to decide: KEEP / REMOVE / REVIEW
    
    KEEP   → high confidence the keyword belongs
    REVIEW → borderline — needs a human to check
    REMOVE → keyword does not match the complaint
    """
    # Encode the complaint
    complaint_emb = embedder.encode(str(complaint_text), convert_to_tensor=True)

    # Similarity scores (0 = totally unrelated, 1 = identical meaning)
    desc_sim    = util.cos_sim(complaint_emb, kw_desc_embeddings[keyword]).item()
    example_sim = util.cos_sim(complaint_emb, kw_example_embeddings[keyword]).item()

    # Combined score — weight example similarity higher (it's more specific)
    combined_sim = (desc_sim * 0.4) + (example_sim * 0.6)

    # Decision
    if combined_sim >= DESC_THRESHOLD + 0.10:
        verdict = 'KEEP'
        reason  = f"Strong match — desc_sim={desc_sim:.3f}, example_sim={example_sim:.3f}"

    elif combined_sim >= DESC_THRESHOLD:
        verdict = 'REVIEW'
        reason  = (
            f"Borderline match — desc_sim={desc_sim:.3f}, "
            f"example_sim={example_sim:.3f}. "
            f"Human review recommended."
        )

    else:
        verdict = 'REMOVE'
        reason  = (
            f"Low similarity — desc_sim={desc_sim:.3f}, "
            f"example_sim={example_sim:.3f}. "
            f"Complaint text does not match this keyword."
        )

    return {
        'verdict'    : verdict,
        'desc_sim'   : round(desc_sim,    4),
        'example_sim': round(example_sim, 4),
        'combined'   : round(combined_sim, 4),
        'reason'     : reason
    }

# ─────────────────────────────────────────────────────────
# RUN VALIDATION ACROSS ENTIRE DATASET
# ─────────────────────────────────────────────────────────
print("Validating keywords for all complaints...")

verified_col = []    # keywords confirmed as correct
removed_col  = []    # keywords removed (mismatch)
review_col   = []    # keywords needing human review
audit_rows   = []    # full audit log

for i, row in df.iterrows():
    complaint_text = str(row.get('clean_summary', row.get('summary', '')))
    tags = row['keywords_list']

    keep, remove, review = [], [], []

    for kw in tags:
        # Skip if keyword not in master list
        if kw not in MASTER_KEYWORDS:
            review.append(kw)
            audit_rows.append({
                'case_reference': row.get('case_reference', i),
                'keyword'       : kw,
                'verdict'       : 'REVIEW',
                'desc_sim'      : None,
                'example_sim'   : None,
                'combined'      : None,
                'reason'        : 'Keyword not in master list — check normalization'
            })
            continue

        result = validate_keyword(complaint_text, kw)

        audit_rows.append({
            'case_reference': row.get('case_reference', i),
            'keyword'       : kw,
            **result
        })

        if result['verdict'] == 'KEEP':
            keep.append(kw)
        elif result['verdict'] == 'REMOVE':
            remove.append(kw)
        else:
            review.append(kw)

    verified_col.append(keep)
    removed_col.append(remove)
    review_col.append(review)

    # Progress every 500 rows
    if i % 500 == 0:
        print(f"  Processed {i}/{len(df)} rows...")

df['verified_keywords'] = verified_col
df['removed_keywords']  = removed_col
df['review_keywords']   = review_col

# ─────────────────────────────────────────────────────────
# RESULTS SUMMARY
# ─────────────────────────────────────────────────────────
audit_df = pd.DataFrame(audit_rows)
verdicts = audit_df['verdict'].value_counts()

print(f"\n{'='*50}")
print(f"VALIDATION SUMMARY")
print(f"{'='*50}")
print(f"  KEEP   : {verdicts.get('KEEP',   0):>6}  ✅  (correct tags, use these)")
print(f"  REVIEW : {verdicts.get('REVIEW', 0):>6}  🔍  (borderline, check manually)")
print(f"  REMOVE : {verdicts.get('REMOVE', 0):>6}  ❌  (wrong tag, drop these)")
print()
print(f"Cases with at least 1 verified keyword : "
      f"{df['verified_keywords'].apply(len).gt(0).sum()}")
print(f"Cases with ALL keywords removed         : "
      f"{df['verified_keywords'].apply(len).eq(0).sum()}")

# ─────────────────────────────────────────────────────────
# EXPORT OUTPUTS
# ─────────────────────────────────────────────────────────

# 1. Clean dataset — only verified keywords
final_df = df[df['verified_keywords'].apply(len) > 0].copy()
final_df['final_keywords'] = final_df['verified_keywords']
final_df.to_csv("validated_complaints.csv", index=False)

# 2. Full audit log — every keyword decision with reason
audit_df.to_csv("keyword_audit_log.csv", index=False)

# 3. Review queue — only REVIEW rows, sorted lowest similarity first
review_queue = audit_df[audit_df['verdict'] == 'REVIEW'].sort_values('combined')
review_queue.to_csv("review_queue.csv", index=False)

print(f"\n📁 Saved:")
print(f"   validated_complaints.csv  — {len(final_df)} clean cases")
print(f"   keyword_audit_log.csv     — {len(audit_df)} keyword decisions")
print(f"   review_queue.csv          — {len(review_queue)} rows to manually check")
