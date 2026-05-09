#Phase 2: Case Shortlisting (3,500–5,000 per Keyword)
# ─── Stratified shortlisting targeting 3500–5000 per keyword ───

TARGET_MIN = 3500
TARGET_MAX = 5000

def shortlist_balanced(df, tag_col='tags_cleaned', target_min=3500, target_max=5000, seed=42):
    np.random.seed(seed)
    df = df.copy().reset_index(drop=True)
    
    tag_counts = Counter(t for tags in df[tag_col] for t in tags)
    all_tags = list(tag_counts.keys())
    
    selected = set()
    keyword_coverage = Counter()
    
    # Score each case: prioritize cases with rarest tags
    def rarity_score(tags):
        return sum(1 / tag_counts[t] for t in tags)
    
    df['_score'] = df[tag_col].apply(rarity_score)
    df_sorted = df.sort_values('_score', ascending=False)
    
    # Pass 1: ensure every keyword hits target_min
    for idx, row in df_sorted.iterrows():
        if any(keyword_coverage[t] < target_min for t in row[tag_col]):
            selected.add(idx)
            for t in row[tag_col]:
                keyword_coverage[t] += 1
    
    # Pass 2: cap at target_max — remove cases where all tags are over target_max
    to_remove = set()
    for idx in list(selected):
        row_tags = df.loc[idx, tag_col]
        if all(keyword_coverage[t] > target_max for t in row_tags):
            # Removing this case won't drop any tag below target_min
            if all(keyword_coverage[t] - 1 >= target_min for t in row_tags):
                to_remove.add(idx)
                for t in row_tags:
                    keyword_coverage[t] -= 1
    
    selected -= to_remove
    result = df.loc[sorted(selected)].drop(columns=['_score']).reset_index(drop=True)
    
    # Report
    final_counts = Counter(t for tags in result[tag_col] for t in tags)
    print(f"\nShortlisted: {len(result)} cases")
    print(f"Keyword coverage (min/max/mean): "
          f"{min(final_counts.values())} / {max(final_counts.values())} / "
          f"{int(np.mean(list(final_counts.values())))}")
    
    # Flag any keywords still below target
    under = {k: v for k, v in final_counts.items() if v < target_min}
    if under:
        print(f"\n⚠️ Keywords still under {target_min} cases (consider oversampling):")
        for k, v in sorted(under.items(), key=lambda x: x[1]):
            print(f"  {k}: {v}")
    
    return result

shortlisted_df = shortlist_balanced(df, tag_col='tags_cleaned')
