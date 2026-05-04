# ═══════════════════════════════════════════════════════════════════
# INTELLIGENT TIER2 DESCRIPTION ENRICHMENT
# PURPOSE: Build rich tag descriptions using real complaint examples
#          FOR CLEANLAB PROBE USE ONLY — never for model training/testing
# ═══════════════════════════════════════════════════════════════════

import pandas as pd
import numpy as np
import json
import re
from pathlib import Path
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer

CACHE_PATH = Path('saved_data/enriched_tag_dictionary.json')


# ── STEP 1: TEXT CLEANING ─────────────────────────────────────────

def clean_text(text):
    """Lowercase, strip special chars, normalise whitespace."""
    text = str(text).lower().strip()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# ── STEP 2: EXTRACT DISCRIMINATIVE KEYWORDS PER TAG ──────────────

def extract_tag_keywords(df_tr, y_tr, valid_tags, top_n=20):
    """
    For each tag, find words that appear MORE in that tag's complaints
    than in the rest of the dataset — true discriminative keywords.
    Uses TF-IDF + frequency contrast (not just raw frequency).
    """
    all_summaries = df_tr['summary'].fillna('').apply(clean_text).tolist()

    # Global word frequencies across ALL complaints
    global_counts = Counter()
    for s in all_summaries:
        global_counts.update(s.split())
    total_docs = len(all_summaries)

    tag_keywords = {}

    for i, tag in enumerate(valid_tags):
        mask       = y_tr[:, i] == 1
        tag_docs   = df_tr.loc[mask, 'summary'].fillna('').apply(clean_text).tolist()
        n_tag_docs = len(tag_docs)

        if n_tag_docs < 3:
            tag_keywords[tag] = []
            continue

        # TF-IDF within this tag's complaints
        tfidf = TfidfVectorizer(
            max_features=2000,
            ngram_range=(1, 2),
            sublinear_tf=True,
            stop_words='english',
            min_df=2
        )
        try:
            X = tfidf.fit_transform(tag_docs)
        except ValueError:
            tag_keywords[tag] = []
            continue

        # Mean TF-IDF score per term within tag
        mean_tfidf = np.array(X.mean(axis=0)).flatten()
        terms      = tfidf.get_feature_names_out()

        # Contrast score: penalise terms that appear everywhere
        # (high in tag + low globally = truly discriminative)
        contrast_scores = []
        for j, term in enumerate(terms):
            global_freq = global_counts.get(term.split()[0], 0) / max(total_docs, 1)
            contrast    = mean_tfidf[j] * (1 - min(global_freq * 5, 0.9))
            contrast_scores.append((term, round(float(mean_tfidf[j]), 4), round(contrast, 4)))

        # Sort by contrast score
        contrast_scores.sort(key=lambda x: x[2], reverse=True)
        tag_keywords[tag] = contrast_scores[:top_n]

    return tag_keywords


# ── STEP 3: EXTRACT BEST REPRESENTATIVE EXAMPLES ─────────────────

def extract_representative_examples(df_tr, y_tr, valid_tags,
                                     tag_keywords, n_examples=3,
                                     max_chars=180):
    """
    For each tag, pick the most representative complaint summaries.
    'Most representative' = contains the highest number of that
    tag's top discriminative keywords.
    """
    tag_examples = {}

    for i, tag in enumerate(valid_tags):
        mask      = y_tr[:, i] == 1
        tag_rows  = df_tr.loc[mask, 'summary'].fillna('').tolist()

        if not tag_rows:
            tag_examples[tag] = []
            continue

        # Get top keywords for this tag
        kws = [kw for kw, _, _ in tag_keywords.get(tag, [])[:15]]
        kw_set = set(' '.join(kws).split())

        # Score each complaint by keyword overlap
        scored = []
        for summary in tag_rows:
            words   = set(clean_text(summary).split())
            overlap = len(words & kw_set)
            scored.append((overlap, summary))

        # Pick top n_examples by keyword overlap
        scored.sort(key=lambda x: x[0], reverse=True)
        best = [s[:max_chars].strip() + '...'
                if len(s) > max_chars else s.strip()
                for _, s in scored[:n_examples]]

        tag_examples[tag] = best

    return tag_examples


# ── STEP 4: BUILD THE ENRICHED DICTIONARY ────────────────────────

def build_enriched_dictionary(valid_tags, tag_keywords,
                               tag_examples, df_tr, y_tr):
    """
    Combines:
      - Tag name
      - Existing tier2_consolidated_description (if available)
      - Top discriminative keywords (unigrams + bigrams)
      - Real complaint examples scored by keyword overlap
    into one rich dictionary entry per tag.
    """
    enriched = {}

    # Get existing descriptions — one per tag (use first available)
    existing_descs = {}
    if 'tier2_consolidated_description' in df_tr.columns:
        for i, tag in enumerate(valid_tags):
            mask = y_tr[:, i] == 1
            descs = df_tr.loc[mask, 'tier2_consolidated_description'].dropna()
            if len(descs) > 0:
                existing_descs[tag] = descs.iloc[0]

    for tag in valid_tags:
        n_complaints = int((y_tr[:, valid_tags.index(tag)] == 1).sum())

        # Keywords — separate unigrams from bigrams
        all_kws   = tag_keywords.get(tag, [])
        unigrams  = [kw for kw, _, _ in all_kws if ' ' not in kw][:10]
        bigrams   = [kw for kw, _, _ in all_kws if ' ' in kw][:8]
        examples  = tag_examples.get(tag, [])
        base_desc = existing_descs.get(tag, '')

        enriched[tag] = {
            'tag':                   tag,
            'n_training_complaints': n_complaints,
            'base_description':      base_desc,
            'top_unigrams':          unigrams,
            'top_bigrams':           bigrams,
            'representative_examples': examples,

            # Pre-built probe text — this is what goes into CleanLab TF-IDF
            'probe_text': (
                f"Tag: {tag}. "
                f"{base_desc} "
                f"Key terms: {', '.join(unigrams[:8])}. "
                f"Key phrases: {', '.join(bigrams[:5])}. "
                f"Examples: {' | '.join(examples)}"
            )
        }

    return enriched


# ── STEP 5: RUN EVERYTHING ────────────────────────────────────────

if CACHE_PATH.exists():
    with open(CACHE_PATH) as f:
        enriched_dict = json.load(f)
    print(f"✓ Loaded cached dictionary — {len(enriched_dict)} tags")

else:
    print("Step 1/3 — Extracting discriminative keywords per tag...")
    tag_keywords = extract_tag_keywords(df_tr, y_tr, valid_tags, top_n=20)

    print("Step 2/3 — Extracting representative examples per tag...")
    tag_examples = extract_representative_examples(
        df_tr, y_tr, valid_tags, tag_keywords,
        n_examples=3, max_chars=180
    )

    print("Step 3/3 — Building enriched dictionary...")
    enriched_dict = build_enriched_dictionary(
        valid_tags, tag_keywords, tag_examples, df_tr, y_tr
    )

    CACHE_PATH.parent.mkdir(exist_ok=True)
    with open(CACHE_PATH, 'w') as f:
        json.dump(enriched_dict, f, indent=2)

    print(f"✓ Saved: {CACHE_PATH}")


# ── STEP 6: PREVIEW THE DICTIONARY ───────────────────────────────

print("\n" + "═" * 65)
print("ENRICHED TAG DICTIONARY — SAMPLE ENTRIES")
print("═" * 65)

for tag, entry in list(enriched_dict.items())[:4]:
    print(f"\n┌─ TAG: {tag}")
    print(f"│  Training complaints: {entry['n_training_complaints']}")
    print(f"│  Base description: {entry['base_description'][:100]}...")
    print(f"│  Top unigrams: {', '.join(entry['top_unigrams'][:8])}")
    print(f"│  Top bigrams:  {', '.join(entry['top_bigrams'][:5])}")
    print(f"│  Examples:")
    for j, ex in enumerate(entry['representative_examples']):
        print(f"│    [{j+1}] {ex}")
    print(f"└─ Probe text preview: {entry['probe_text'][:200]}...")


# ── STEP 7: BUILD CLEANLAB INPUT TEXT ────────────────────────────
# Attach probe_text per tag to each complaint row
# ONLY used as CleanLab TF-IDF input — never in model features

def build_cleanlab_input(row, y_row, enriched_dict, valid_tags):
    """
    For each complaint row, append the probe_text of all its tags
    to the base complaint fields.
    FOR CLEANLAB USE ONLY.
    """
    service = str(row.get('Service', '')).strip()
    tier1   = str(row.get('tier1', '')).strip()
    summary = str(row.get('summary', ''))[:250]

    tag_probes = []
    for i, tag in enumerate(valid_tags):
        if y_row[i] == 1:
            probe = enriched_dict.get(tag, {}).get('probe_text', tag)
            tag_probes.append(probe)

    return (
        f"{service} | {tier1} | {summary} | "
        + " || ".join(tag_probes)
    )


print("\nBuilding CleanLab input text for all training rows...")
cleanlab_texts = [
    build_cleanlab_input(df_tr.iloc[i], y_tr[i], enriched_dict, valid_tags)
    for i in range(len(df_tr))
]

print(f"✓ {len(cleanlab_texts)} rows ready for CleanLab TF-IDF")
print(f"\nSample CleanLab input (first 500 chars):")
print(cleanlab_texts[0][:500])

# Save for reference / reproducibility
pd.DataFrame({
    'case_reference':   df_tr['case_reference'].values,
    'cleanlab_input':   cleanlab_texts
}).to_csv('saved_data/cleanlab_input_enriched.csv', index=False)

print("\n✓ Saved: saved_data/cleanlab_input_enriched.csv")
print("→ Pass 'cleanlab_texts' to TfidfVectorizer in Block 5")
