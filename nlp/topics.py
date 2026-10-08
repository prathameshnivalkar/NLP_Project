from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def detect_topics(
    texts: list[str],
    n_topics: int = 4,
    top_words: int = 5,
) -> list[dict]:
    if len(texts) < n_topics:
        n_topics = max(1, len(texts))

    try:
        vectorizer = CountVectorizer(
            max_df=0.95, min_df=2, max_features=1000, stop_words="english"
        )
        doc_term = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out()

        lda = LatentDirichletAllocation(
            n_components=n_topics, random_state=42, max_iter=20
        )
        lda.fit(doc_term)

        topics: list[dict] = []
        for idx, topic_vec in enumerate(lda.components_):
            top_indices = topic_vec.argsort()[-top_words:][::-1]
            words = [feature_names[i] for i in top_indices]
            topics.append(
                {
                    "name": " ".join(words[:3]).title(),
                    "share": round(topic_vec.sum()),
                    "tone": "neutral",
                    "detail": ", ".join(words),
                }
            )
        return topics

    except ValueError:
        return _fallback_topics(texts, n_topics, top_words)


def _fallback_topics(
    texts: list[str], n_topics: int, top_words: int
) -> list[dict]:
    all_words: list[str] = []
    for t in texts:
        all_words.extend(t.split())

    from collections import Counter

    common = Counter(all_words).most_common(n_topics * top_words)
    topics: list[dict] = []
    for i in range(n_topics):
        start = i * top_words
        end = start + top_words
        words = [w for w, _ in common[start:end]]
        topics.append(
            {
                "name": " ".join(words[:3]).title() if words else f"Topic {i+1}",
                "share": len(texts) // (n_topics or 1),
                "tone": "neutral",
                "detail": ", ".join(words) if words else "no data",
            }
        )
    return topics
