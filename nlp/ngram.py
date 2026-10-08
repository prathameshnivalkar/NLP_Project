"""Experiment 05 - N-Gram Language Model.

Counts sequences of N consecutive words across the comment corpus so the
analyzer can talk about phrases such as ``audio quality`` instead of isolated
words. Bigrams and trigrams are what turn keyword counts into discussion
signals, and the same counts bucketed over time are what the trend panel plots.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone

from .preprocessor import SENTENCE_BOUNDARY

NGRAM_SIZES = (1, 2, 3)
NGRAM_LABELS: dict[int, str] = {1: "Unigram", 2: "Bigram", 3: "Trigram"}
TREND_BUCKETS = 8


def _crosses_sentence(gram: tuple[str, ...]) -> bool:
    return SENTENCE_BOUNDARY in gram


def generate_ngrams(tokens: list[str], size: int) -> list[tuple[str, ...]]:
    """Every sequence of ``size`` consecutive tokens, in order.

    Sequences that straddle a sentence boundary are dropped, so a bigram is
    always a phrase the author actually wrote.
    """
    if size < 1 or len(tokens) < size:
        return []
    return [
        gram
        for gram in (tuple(tokens[i : i + size]) for i in range(len(tokens) - size + 1))
        if not _crosses_sentence(gram)
    ]


def count_ngrams(token_lists: list[list[str]], size: int) -> Counter[tuple[str, ...]]:
    """Occurrence count of every n-gram of the given size in the corpus."""
    counts: Counter[tuple[str, ...]] = Counter()
    for tokens in token_lists:
        counts.update(generate_ngrams(tokens, size))
    return counts


def _is_repetitive(gram: tuple[str, ...]) -> bool:
    """Drop fillers such as ``very very`` and ``so so good``."""
    return len(gram) > 1 and len(set(gram)) == 1


def top_ngrams(
    token_lists: list[list[str]],
    size: int,
    top_k: int = 6,
    min_count: int = 2,
) -> list[dict]:
    """Most frequent n-grams, scored by count then by document spread.

    Document spread matters for trend detection: a phrase carried by many
    comments is a discussion, the same phrase repeated in one comment is noise.
    """
    if not token_lists or size < 1:
        return []

    counts = count_ngrams(token_lists, size)
    if not counts:
        return []

    spread: Counter[tuple[str, ...]] = Counter()
    for tokens in token_lists:
        spread.update(set(generate_ngrams(tokens, size)))

    def rank(threshold: int) -> list[tuple[tuple[str, ...], int, int]]:
        return sorted(
            (
                (gram, count, spread[gram])
                for gram, count in counts.items()
                if count >= threshold and not _is_repetitive(gram)
            ),
            key=lambda item: (-item[1], -item[2], item[0]),
        )

    # ``min_count`` is a preference, not a hard floor: short corpora repeat no
    # phrase often, so fall back to single occurrences rather than an empty panel.
    ranked = rank(min_count) or rank(1)
    top = ranked[:top_k]
    if not top:
        return []

    top_count = top[0][1] or 1
    return [
        {
            "phrase": " ".join(gram),
            "size": size,
            "label": NGRAM_LABELS.get(size, f"{size}-gram"),
            "count": count,
            "documents": documents,
            "share": round(count / top_count * 100),
        }
        for gram, count, documents in top
    ]


def _parse_timestamp(value: str) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _assign_buckets(records: list[dict], buckets: int) -> list[int]:
    """Map each record to a time bucket, oldest first.

    Comments arrive newest-first from the API, so they are sorted by publish
    time. When no timestamps are available the corpus is split into equal
    sequential buckets instead, which keeps the chart populated.
    """
    times = [_parse_timestamp(record.get("published_at", "")) for record in records]
    if all(times):
        ordered = sorted(range(len(records)), key=lambda i: times[i])
        start, end = times[ordered[0]], times[ordered[-1]]
        span = (end - start).total_seconds() or 1.0
        return [
            min(buckets - 1, int((times[i] - start).total_seconds() / span * buckets))
            for i in range(len(records))
        ]

    size = max(1, len(records) // buckets)
    return [min(buckets - 1, i // size) for i in range(len(records))]


def phrase_trends(
    records: list[dict],
    top_k: int = 4,
    buckets: int = TREND_BUCKETS,
) -> list[dict]:
    """Track the leading bigrams over time to surface trending discussions.

    ``records`` is one dict per comment with ``tokens`` (already lemmatized),
    an optional ``published_at`` and an optional ``sentiment`` label.
    """
    token_lists = [record.get("tokens") or [] for record in records]
    if not any(token_lists) or buckets < 2:
        return []

    leaders = top_ngrams(token_lists, size=2, top_k=top_k, min_count=2)
    if not leaders:
        leaders = top_ngrams(token_lists, size=2, top_k=top_k, min_count=1)
    if not leaders:
        return []

    bucket_of = _assign_buckets(records, buckets)
    per_bucket: dict[str, Counter[int]] = {
        leader["phrase"]: Counter() for leader in leaders
    }
    tone: dict[str, list[str]] = {leader["phrase"]: [] for leader in leaders}
    doc_hits: Counter[str] = Counter()

    for record, index in zip(records, bucket_of):
        grams = {" ".join(gram) for gram in generate_ngrams(record.get("tokens") or [], 2)}
        sentiment = record.get("sentiment")
        for phrase in grams & per_bucket.keys():
            per_bucket[phrase][index] += 1
            doc_hits[phrase] += 1
            if sentiment:
                tone[phrase].append(sentiment)

    trends: list[dict] = []
    for leader in leaders:
        phrase = leader["phrase"]
        series = [per_bucket[phrase].get(i, 0) for i in range(buckets)]
        total = sum(series)
        if not total:
            continue

        head = max(1, len(series) // 3)
        early = sum(series[:head]) or 0
        late = sum(series[-head:])
        momentum = (late - early) / max(early, 1)
        sentiments = tone[phrase]
        positive = sentiments.count("positive")
        negative = sentiments.count("negative")
        if positive + negative == 0:
            stance = "neutral"
        elif positive >= negative * 2:
            stance = "positive"
        elif negative >= positive * 2:
            stance = "negative"
        else:
            stance = "mixed"

        trends.append(
            {
                "phrase": phrase,
                "size": 2,
                "count": total,
                "documents": doc_hits[phrase],
                "series": series,
                "momentum": round(momentum, 2),
                "direction": "rising" if momentum > 0.2 else "cooling" if momentum < -0.2 else "steady",
                "stance": stance,
            }
        )

    trends.sort(key=lambda row: (-row["documents"], -row["count"]))
    return trends


def ngram_report(
    token_lists: list[list[str]],
    top_k: int = 6,
    min_count: int = 2,
) -> dict:
    """Unigram, bigram and trigram frequency tables for the dashboard."""
    if not token_lists or not any(token_lists):
        return _empty_report()

    tables = {
        NGRAM_LABELS[size]: top_ngrams(token_lists, size, top_k=top_k, min_count=min_count)
        for size in NGRAM_SIZES
    }

    documents = [tokens for tokens in token_lists if tokens]
    phrase_hits = 0
    for tokens in documents:
        grams = {" ".join(gram) for gram in generate_ngrams(tokens, 2)}
        if grams & {row["phrase"] for row in tables["Bigram"]}:
            phrase_hits += 1

    total_tokens = sum(len(tokens) for tokens in documents)
    return {
        "documents": len(documents),
        "total_tokens": total_tokens,
        "total_grams": sum(
            len(generate_ngrams(tokens, size))
            for size in NGRAM_SIZES
            for tokens in documents
        ),
        "unigrams": tables["Unigram"],
        "bigrams": tables["Bigram"],
        "trigrams": tables["Trigram"],
        "phrase_coverage": round(phrase_hits / len(documents) * 100, 1) if documents else 0.0,
    }


def _empty_report() -> dict:
    return {
        "documents": 0,
        "total_tokens": 0,
        "total_grams": 0,
        "unigrams": [],
        "bigrams": [],
        "trigrams": [],
        "phrase_coverage": 0.0,
    }


__all__ = [
    "NGRAM_LABELS",
    "count_ngrams",
    "generate_ngrams",
    "ngram_report",
    "phrase_trends",
    "top_ngrams",
]
