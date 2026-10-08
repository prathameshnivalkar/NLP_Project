from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer


def extract_keywords(texts: list[str], top_n: int = 10) -> list[dict]:
    """Rank single words by TF-IDF.

    Unigrams only: Experiment 05 owns the multi-word phrases, and the input has
    already been root-normalized by Experiment 04 so that "tutorial" and
    "tutorials" compete as one term.
    """
    if not texts:
        return []

    vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 1))
    try:
        tfidf_matrix = vectorizer.fit_transform(texts)
    except ValueError:
        return simple_keywords(texts, top_n)

    feature_names = vectorizer.get_feature_names_out()
    sums = tfidf_matrix.toarray().sum(axis=0)
    scored = list(zip(feature_names, sums))
    scored.sort(key=lambda x: x[1], reverse=True)

    if not scored:
        return []

    max_score = scored[0][1] or 1
    return [
        {"label": word, "weight": round(score / max_score * 100)}
        for word, score in scored[:top_n]
    ]


def simple_keywords(texts: list[str], top_n: int = 8) -> list[dict]:
    word_counts: Counter[str] = Counter()
    for text in texts:
        for word in text.split():
            if len(word) > 2:
                word_counts[word] += 1
    if not word_counts:
        return []
    most_common = word_counts.most_common(top_n)
    max_count = most_common[0][1] or 1
    return [
        {"label": word, "weight": round(count / max_count * 100)}
        for word, count in most_common
    ]
