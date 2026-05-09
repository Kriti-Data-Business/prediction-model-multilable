#Phase 1: Keyword Cleaning (96 Tags)

import pandas as pd
import numpy as np
from rapidfuzz import process, fuzz
import re
from collections import Counter

# ─── Your master keyword list (ground truth) ───
MASTER_KEYWORDS = [
    "billing issue", "refund request", "delivery delay",
    # ... all 96 keywords
]

def normalize_keyword(kw):
    """Lowercase, strip extra spaces, remove stray punctuation."""
    kw = kw.lower().strip()
    kw = re.sub(r'[^\w\s]', '', kw)       # remove punctuation
    kw = re.sub(r'\s+', ' ', kw)           # collapse spaces
    return kw

def fuzzy_match_keyword(raw_kw, master_list, threshold=85):
    """Map a raw human-typed keyword to the closest master keyword."""
    raw_clean = normalize_keyword(raw_kw)
    match, score, _ = process.extractOne(
        raw_clean,
        [normalize_keyword(k) for k in master_list],
        scorer=fuzz.token_sort_ratio
    )
    if score >= threshold:
        return master_list[[normalize_keyword(k) for k in master_list].index(match)]
    return None  # no confident match — flag for review

# Apply to your dataset
# df['tags'] is a list column e.g. ['billingg issue', 'refund  request']
df['tags_cleaned'] = df['tags'].apply(
    lambda tags: list(filter(None, [fuzzy_match_keyword(t, MASTER_KEYWORDS) for t in tags]))
)

# Audit: keywords that didn't match anything
df['unmatched_tags'] = df['tags'].apply(
    lambda tags: [t for t in tags if fuzzy_match_keyword(t, MASTER_KEYWORDS) is None]
)
print(Counter(tag for tags in df['unmatched_tags'] for tag in tags))
