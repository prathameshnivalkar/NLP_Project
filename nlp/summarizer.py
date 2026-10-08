from textblob import TextBlob


def summarize_comments(texts: list[str], max_sentences: int = 3) -> str:
    if not texts:
        return "No comments to summarize."

    scored: list[tuple[str, float]] = []
    for text in texts:
        blob = TextBlob(text)
        score = len(blob.sentences) + blob.sentiment.polarity
        scored.append((text, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    top = [t for t, _ in scored[: max_sentences * 3]]

    combined = " ".join(top)
    blob = TextBlob(combined)
    sentences = blob.sentences

    if not sentences:
        return combined[:300]

    summary_sentences = sentences[:max_sentences]
    return " ".join(str(s) for s in summary_sentences)
