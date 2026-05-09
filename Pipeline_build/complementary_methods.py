from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Define sentiment expectations per keyword
# e.g. 'refund request' → typically negative, 'resolved complaint' → positive
KEYWORD_SENTIMENT_MAP = {
    "billing issue": "negative",
    "refund request": "negative",
    "delivery delay": "negative",
    "positive feedback": "positive",
    # ... map all 96 keywords
}

analyzer = SentimentIntensityAnalyzer()

def check_sentiment_label_conflict(text, tags, keyword_sentiment_map):
    score = analyzer.polarity_scores(text)['compound']
    sentiment = "positive" if score > 0.05 else "negative" if score < -0.05 else "neutral"
    
    conflicts = []
    for tag in tags:
        expected = keyword_sentiment_map.get(tag)
        if expected and expected != "neutral" and expected != sentiment:
            conflicts.append((tag, expected, sentiment))
    return conflicts

df['sentiment_conflicts'] = df.apply(
    lambda r: check_sentiment_label_conflict(r['text'], r['tags_cleaned'], KEYWORD_SENTIMENT_MAP),
    axis=1
)

# Cases with conflicts — review or relabel
conflict_df = df[df['sentiment_conflicts'].apply(len) > 0]
print(f"Cases with sentiment-label conflicts: {len(conflict_df)}")


#Method A — Sentiment-Keyword Consistency Check (NLP-based)
from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Define sentiment expectations per keyword
# e.g. 'refund request' → typically negative, 'resolved complaint' → positive
KEYWORD_SENTIMENT_MAP = {
    "billing issue": "negative",
    "refund request": "negative",
    "delivery delay": "negative",
    "positive feedback": "positive",
    # ... map all 96 keywords
}

analyzer = SentimentIntensityAnalyzer()

def check_sentiment_label_conflict(text, tags, keyword_sentiment_map):
    score = analyzer.polarity_scores(text)['compound']
    sentiment = "positive" if score > 0.05 else "negative" if score < -0.05 else "neutral"
    
    conflicts = []
    for tag in tags:
        expected = keyword_sentiment_map.get(tag)
        if expected and expected != "neutral" and expected != sentiment:
            conflicts.append((tag, expected, sentiment))
    return conflicts

df['sentiment_conflicts'] = df.apply(
    lambda r: check_sentiment_label_conflict(r['text'], r['tags_cleaned'], KEYWORD_SENTIMENT_MAP),
    axis=1
)

# Cases with conflicts — review or relabel
conflict_df = df[df['sentiment_conflicts'].apply(len) > 0]
print(f"Cases with sentiment-label conflicts: {len(conflict_df)}")



