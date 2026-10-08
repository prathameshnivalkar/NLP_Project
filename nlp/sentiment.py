from textblob import TextBlob


def analyze_sentiment(text: str) -> str:
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.1:
        return "positive"
    if polarity < -0.1:
        return "negative"
    return "neutral"


def sentiment_batch(texts: list[str]) -> list[dict]:
    results = {"positive": 0, "negative": 0, "neutral": 0}
    labeled: list[dict] = []
    for text in texts:
        label = analyze_sentiment(text)
        results[label] += 1
        labeled.append({"text": text, "sentiment": label})
    total = len(texts) or 1
    return {
        "counts": results,
        "percentages": {
            k: round(v / total * 100) for k, v in results.items()
        },
        "labeled": labeled,
    }
