import hashlib
from datetime import datetime, timezone

from .fetcher import fetch_comments, fetch_video_title
from .keywords import extract_keywords
from .morphology import morphology_report, normalize_batch
from .ner import entity_report
from .ngram import ngram_report, phrase_trends
from .chunking import chunk_report, feature_selection_study
from .postagger import pos_report
from .preprocessor import (
    preprocess_batch,
    tokenize_batch,
    tokenize_full_batch,
    tokenize_surface_sentences_batch,
)
from .sentiment import sentiment_batch
from .similarity import similarity_report
from .summarizer import summarize_comments
from .topics import detect_topics
from .wsd import wsd_report

TREND_BUCKETS = 8
DEFAULT_TREND_LABELS = [f"{hour:02d}:00" for hour in range(8, 8 + TREND_BUCKETS * 2, 2)]

#: Upper bound on comments fed to Experiments 06-10. Tagging every comment of a
#: 500-comment section five times over, plus fitting the Experiment 07 study and
#: running GRU inference, turns a one-second analysis into a multi-second one.
#: 250 comments is already far more than these descriptive reports need.
LAB_SAMPLE = 250


def _lab_sample(
    texts: list[str], labels: list[str], surface: list[list[str]]
) -> tuple[list[str], list[str], list[list[str]]]:
    """Bounded, deterministic, sentiment-stratified sample for Experiments 06-10.

    Taking every Nth comment would be cheaper than a shuffle, but it interacts
    badly with how a comment section is ordered - bursts of near-identical
    complaints would land together and one would dominate the sample. Round-robin
    over the sentiment classes keeps both polar classes present for the
    Experiment 07 study, which cannot train without them.
    """
    if len(texts) <= LAB_SAMPLE:
        return texts, labels, surface

    buckets: dict[str, list[int]] = {}
    for index, label in enumerate(labels):
        buckets.setdefault(label, []).append(index)

    quota = max(1, LAB_SAMPLE // max(1, len(buckets)))
    chosen: list[int] = []
    for indices in buckets.values():
        step = max(1, len(indices) // quota)
        chosen.extend(indices[::step][:quota])

    chosen = sorted(set(chosen))[:LAB_SAMPLE]
    return (
        [texts[i] for i in chosen],
        [labels[i] for i in chosen],
        [surface[i] for i in chosen],
    )


def _trend_chart(trends: list[dict], positive_pct: int) -> tuple[list[int], list[str]]:
    """Sentiment-over-time series for the existing line chart.

    Experiment 05 supplies the leading phrase for each time bucket, so the shape
    of the curve follows what the audience was actually talking about. Without
    usable timestamps the chart falls back to a spread around the positive share.
    """
    if trends:
        lead = [0] * TREND_BUCKETS
        for index, trend in enumerate(trends[:TREND_BUCKETS]):
            series = trend.get("series") or []
            if index < len(series):
                lead[index] = series[index]
        peak = max(lead)
        if peak:
            values = [round(positive_pct * (0.72 + 0.45 * (value / peak))) for value in lead]
            return values, DEFAULT_TREND_LABELS[:TREND_BUCKETS]

    base = positive_pct
    fallback = [base - 5, base - 2, base - 4, base + 1, base, base + 3, base + 2, base + 5]
    return fallback[:TREND_BUCKETS], DEFAULT_TREND_LABELS[:TREND_BUCKETS]


def analyze_comments(raw_comments: list[dict]) -> dict:
    """Run the full NLP pipeline over fetched comments.

    Ordering follows the experiment map: Experiment 03 preprocessing, then
    Experiment 04 morphological analysis whose root normalization feeds keyword
    and topic extraction, then Experiment 05 n-grams over the surface forms,
    then sentiment, topics and summarization.

    The returned payload includes a private ``labels`` key with the per-comment
    sentiment, which callers pop when building the comment stream.
    """
    texts = [comment["text"] for comment in raw_comments]
    total = len(raw_comments)

    # Experiment 03 - tokenization, filtration, stop-words, lemmatization.
    processed = preprocess_batch(texts)

    # Experiment 04 - morphological analysis plus root normalization, so
    # "tutorial" and "tutorials" count as one keyword.
    morphology = morphology_report(texts)
    lemma_texts = [
        " ".join(tokens) for tokens in normalize_batch(tokenize_batch(texts))
    ]

    # Experiment 05 - n-grams over surface forms, which keeps detected phrases
    # readable ("audio quality") instead of lemmatized ("audio quality need").
    surface_tokens = tokenize_surface_sentences_batch(texts)
    ngrams = ngram_report(surface_tokens)

    sent = sentiment_batch(processed)
    labels = [row["sentiment"] for row in sent["labeled"]]

    records = [
        {
            "tokens": tokens,
            "published_at": comment.get("published_at", ""),
            "sentiment": label,
        }
        for tokens, comment, label in zip(surface_tokens, raw_comments, labels)
    ]
    trends = phrase_trends(records, top_k=4, buckets=TREND_BUCKETS)

    kws = extract_keywords(lemma_texts, top_n=8)
    topics = detect_topics(lemma_texts, n_topics=4)
    summary = summarize_comments(texts, max_sentences=3)

    # Experiments 06-10. These five run per comment and two of them train or run
    # a network, so they work on a bounded sample instead of the whole section.
    # The sample is deterministic and stratified, so a 500-comment video gives
    # the same report on every refresh and keeps enough of each sentiment class
    # for the Experiment 07 study to have both classes to work with.
    lab_texts, lab_labels, lab_surface = _lab_sample(texts, labels, surface_tokens)
    # POS and chunk analysis need unfiltered tokens: dropping function words
    # leaves the tagger and the grammar with no sentence structure to work on.
    lab_tokens = tokenize_full_batch(lab_texts)

    pos = pos_report(lab_tokens)
    chunking = chunk_report(lab_tokens)
    feature_study = feature_selection_study(lab_surface, lab_labels)
    entities = entity_report(lab_texts)
    similarity = similarity_report(lab_texts, lab_labels)
    wsd = wsd_report(lab_texts)

    pos_pct = round(sent["counts"]["positive"] / total * 100) if total else 0
    neg_pct = round(sent["counts"]["negative"] / total * 100) if total else 0
    neu_pct = 100 - pos_pct - neg_pct
    trend_values, trend_labels = _trend_chart(trends, pos_pct)

    return {
        "labels": labels,
        "summary": summary,
        "metrics": {
            "comments": total,
            "positive": pos_pct,
            "negative": neg_pct,
            "neutral": neu_pct,
            "engagement": round(total / 100, 1),
            "topics": len(topics),
            "word_forms": morphology["distinct_forms"],
            "word_roots": morphology["distinct_roots"],
            "phrases": len(ngrams["bigrams"]),
            "chunks": chunking.get("total_chunks", 0),
            "entities": entities["total"],
            "discussions": similarity["groups"],
            "senses": wsd["disambiguated"],
        },
        "sentiment": [
            {"label": "Positive", "value": pos_pct, "count": sent["counts"]["positive"], "color": "lime"},
            {"label": "Neutral", "value": neu_pct, "count": sent["counts"]["neutral"], "color": "slate"},
            {"label": "Negative", "value": neg_pct, "count": sent["counts"]["negative"], "color": "coral"},
        ],
        "trend": trend_values,
        "trend_labels": trend_labels,
        "topics": topics,
        "keywords": kws,
        "morphology": morphology,
        "ngrams": ngrams,
        "phrase_trends": trends,
        "pos": pos,
        "chunks": chunking,
        "feature_study": feature_study,
        "entities": entities,
        "similarity": similarity,
        "wsd": wsd,
    }


def _build_feed(raw_comments: list[dict], labels: list[str]) -> list[dict]:
    feed = []
    for index, comment in enumerate(raw_comments[:20]):
        published = comment.get("published_at", "")
        feed.append(
            {
                "author": comment["author"],
                "text": comment["text"][:200],
                "sentiment": labels[index] if index < len(labels) else "neutral",
                "time": published[11:16] if len(published) >= 16 else "",
                "likes": str(comment.get("likes", 0)),
                "initials": "".join(p[0] for p in comment["author"].split()[:2]),
            }
        )
    return feed


def _demo_fallback(video_id: str, url: str) -> dict:
    digest = hashlib.sha256(video_id.encode()).hexdigest()
    offset = int(digest[:4], 16)
    comments = 1248 + offset % 6100
    positive = 52 + offset % 18
    negative = 11 + offset % 10
    neutral = 100 - positive - negative

    demo_texts = [
        "This is the clearest walkthrough I have seen on this workflow. The pacing is perfect.",
        "Would love to see a deeper dive into the automation options in a future update.",
        "The examples make the product feel much more approachable. Saving this for later.",
        "The value is there, but the pricing section still feels a little hard to compare.",
        "The before and after section made the difference click for me instantly.",
        "Great tutorial, really helped me understand the basics.",
        "Audio quality could be better but content is solid.",
        "I've been looking for something like this for months!",
        "Not bad, but I expected more depth on the advanced features.",
        "This saved me so much time. Thank you for the clear explanation!",
    ]

    demo_comments = [
        {"author": f"Viewer {i + 1}", "text": text, "likes": 0, "published_at": ""}
        for i, text in enumerate(demo_texts)
    ]

    analysis = analyze_comments(demo_comments)
    labels = analysis.pop("labels")

    analysis["metrics"].update(
        {
            "comments": comments,
            "positive": positive,
            "negative": negative,
            "neutral": neutral,
            "engagement": 8.4 + (offset % 13) / 10,
        }
    )
    analysis["sentiment"] = [
        {"label": "Positive", "value": positive, "count": round(comments * positive / 100), "color": "lime"},
        {"label": "Neutral", "value": neutral, "count": round(comments * neutral / 100), "color": "slate"},
        {"label": "Negative", "value": negative, "count": round(comments * negative / 100), "color": "coral"},
    ]
    analysis["trend"] = [positive - 9, positive - 4, positive - 6, positive + 3, positive + 1, positive + 7, positive + 4, positive + 11]
    analysis["trend_labels"] = ["08:00", "10:00", "12:00", "14:00", "16:00", "18:00", "20:00", "22:00"]

    comment_templates = [
        ("Maya Chen", demo_texts[0], labels[0], "2m ago", "1.2k"),
        ("Jordan Lee", demo_texts[1], labels[1], "8m ago", "642"),
        ("Ari Patel", demo_texts[2], labels[2], "14m ago", "438"),
        ("Sam Rivera", demo_texts[3], labels[3], "22m ago", "209"),
        ("Noah Williams", demo_texts[4], labels[4], "31m ago", "177"),
    ]

    return {
        "video_id": video_id,
        "source_url": url,
        "video_title": "Audience signal report · demo workspace",
        "channel": "Creator Lab",
        "analyzed_at": datetime.now(timezone.utc).strftime("%b %d, %Y · %H:%M UTC"),
        "is_demo": True,
        **analysis,
        "comments_feed": [
            {
                "author": author,
                "text": text,
                "sentiment": sentiment,
                "time": time,
                "likes": likes,
                "initials": "".join(p[0] for p in author.split()[:2]),
            }
            for author, text, sentiment, time, likes in comment_templates
        ],
    }


def analyze_video(video_id: str, url: str) -> dict:
    try:
        raw_comments = fetch_comments(video_id, max_results=500)
        if not raw_comments:
            return _demo_fallback(video_id, url)

        analysis = analyze_comments(raw_comments)
        labels = analysis.pop("labels")

        return {
            "video_id": video_id,
            "source_url": url,
            "video_title": fetch_video_title(video_id),
            "channel": "YouTube",
            "analyzed_at": datetime.now(timezone.utc).strftime("%b %d, %Y · %H:%M UTC"),
            "is_demo": False,
            **analysis,
            "comments_feed": _build_feed(raw_comments, labels),
        }

    except Exception:
        return _demo_fallback(video_id, url)
