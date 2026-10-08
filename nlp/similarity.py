"""Experiment 09 - Real-Time Text Similarity Recognizer.

Comment sections repeat themselves. The documentation's own example is the
point: "The audio quality is very poor", "The sound quality is bad" and "I
cannot hear the audio properly" share almost no vocabulary, yet they are one
opinion. This module collapses them into a single discussion so the dashboard
counts distinct opinions instead of repetitions.

Three ideas make that work, because plain bag-of-words cosine does not:

1.  Lemmatisation (Experiment 04 machinery, backed by WordNet) so "tutorial"
    and "tutorials" cannot land in different groups.
2.  Function words are dropped. "the" and "is" are present in every comment and
    would otherwise push unrelated comments towards each other.
3.  Synonym folding for two closed classes that dominate YouTube comments:
    evaluative adjectives (poor/bad/terrible) and auditory words
    (audio/sound/hear). WordNet already links audio to sound, but nothing
    links "poor" to "bad", because WordNet's first sense of "poor" is *poor
    people*. A small curated map is more honest than a bad automatic guess.

A comment joins a group only when it shares a *subject* word with the group and
clears the cosine bar. Both tests are needed: cosine alone joins any two
complaints that both contain the word "bad", while subject overlap alone is too
coarse to be useful.

Grouping is greedy online assignment: each new comment is scored against
existing group centroids and joins the closest one above a threshold, or opens
a new group. That is single-sweep and order-stable, which suits a stream of
comments arriving from the API.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .preprocessor import SENTENCE_BOUNDARY, ensure_nltk, tokenize_full

#: Cosine bar for joining a group. Deliberately low: the topic-overlap test in
#: :func:`group_similar` already rejects off-topic comments, and a high bar would
#: miss paraphrases, which is the whole point of the experiment.
DEFAULT_THRESHOLD = 0.30
MIN_GROUP_SIZE = 2
LABEL_TERMS = 2

#: Evaluative vocabulary is effectively a closed class in comment data, and
#: "very poor" has to meet "bad". Mapped to one canonical token per group.
EVALUATIVE_SYNONYMS = {
    "poor": "bad", "bad": "bad", "terrible": "bad", "awful": "bad",
    "horrible": "bad", "dreadful": "bad", "atrocious": "bad", "worst": "bad",
    "useless": "bad", "broken": "bad", "wrong": "bad", "hate": "bad",
    "disappointed": "bad", "waste": "bad",
    "good": "good", "great": "good", "excellent": "good", "nice": "good",
    "helpful": "good", "useful": "good", "amazing": "good", "best": "good",
    "love": "good", "fantastic": "good", "solid": "good", "clear": "good",
}

#: Everything the viewer is reacting to when they complain about how a video
#: sounds. Keeps an audio complaint and a "cannot hear it" complaint together.
AUDITORY_SYNONYMS = {
    "audio": "audio", "sound": "audio", "sounds": "audio", "sounding": "audio",
    "hear": "audio", "heard": "audio", "hearing": "audio", "listening": "audio",
    "audible": "audio", "noise": "audio", "noisy": "audio", "volume": "audio",
}

#: Canonical tokens that express an opinion rather than a subject. Two comments
#: about unrelated things can both be "bad"; that alone must not join them.
EVALUATIVE_CANONICAL = frozenset({"bad", "good"})

STOPWORDS = frozenset(
    """
    a an the this that these those is are was were be been being am
    i you he she it we they me him her us them my your his its our their
    of in on at to for with about from by as into over after before
    and or but so if when while very really quite just too also more most
    do does did doing done can could should would will shall may might must
    not no nor only own same than then there here what which who whom how why
    have has had having get got make made go went come came
    """.split()
)


def _build_lemmatizer():
    from nltk.stem import WordNetLemmatizer

    return WordNetLemmatizer()


_lemmatizer = _build_lemmatizer()


@lru_cache(maxsize=8192)
def _lemma(word: str) -> str:
    """Part-of-speech aware singular form.

    Deliberately conservative. An earlier version took the first lemma of the
    first synset, which turned "quality" into "choice" and "poor" into
    "poor_people" because WordNet's noun senses of those words are about
    something else entirely. Multi-word synset lemmas are rejected for the same
    reason: they signal the wrong sense was selected.
    """
    ensure_nltk()
    for pos in ("n", "v", "a"):
        candidate = _lemmatizer.lemmatize(word, pos)
        if candidate != word and "_" not in candidate:
            return candidate
    return word


def topic_terms(tokens: list[str]) -> set[str]:
    """The subject words in a feature list, minus bare opinion words."""
    return {token for token in tokens if token not in EVALUATIVE_CANONICAL}


def feature_tokens(text: str) -> list[str]:
    """Content words, lemmatised and synonym-folded, in order of appearance.

    The stop-word test runs on the raw word *and* its lemma: lemmatising first
    turns "which" into "choice", which would smuggle a function word into the
    vocabulary as a topic.
    """
    ensure_nltk()
    tokens: list[str] = []
    for sentence in text.split(SENTENCE_BOUNDARY):
        for word in tokenize_full(sentence):
            lemma = _lemma(word)
            if word in STOPWORDS or lemma in STOPWORDS:
                continue
            canonical = EVALUATIVE_SYNONYMS.get(lemma) or AUDITORY_SYNONYMS.get(lemma)
            tokens.append(canonical or lemma)
    return tokens


def build_features(texts: list[str]) -> list[str]:
    return [" ".join(feature_tokens(text)) for text in texts]


def _label_for_group(texts: list[str], members: list[int]) -> str:
    """Name the group after the subjects it discusses.

    Derived from the members rather than the first comment, so a group of
    complaints is called "Audio Quality" instead of "The Audio Quality Is Very".
    Opinion words are excluded - "Audio Bad" describes a mood, not a topic - and
    only used to pad the label when a group has no subject at all.

    Equally frequent subjects are ordered by where they first appear, so a
    comment about "audio quality" is labelled "Audio Quality" rather than
    "Quality Audio" when both words occur once.
    """
    subjects: Counter = Counter()
    opinions: Counter = Counter()
    first_seen: dict[str, int] = {}
    for position, index in enumerate(members):
        tokens = feature_tokens(texts[index])
        subject_terms = topic_terms(tokens)
        subjects.update(subject_terms)
        opinions.update(token for token in tokens if token not in subject_terms)
        for token in subject_terms:
            if token not in first_seen:
                first_seen[token] = position

    def ranked(counter: Counter) -> list[str]:
        return [
            word
            for word, _ in sorted(
                counter.items(),
                key=lambda item: (
                    -item[1],
                    first_seen.get(item[0], len(members)),
                    item[0],
                ),
            )
        ]

    if not subjects:
        if not opinions:
            return "General"
        return " ".join(word.title() for word, _ in opinions.most_common(LABEL_TERMS))

    label = ranked(subjects)[:LABEL_TERMS]
    if len(label) < LABEL_TERMS and opinions:
        for word in ranked(opinions):
            if word not in label:
                label.append(word)
            if len(label) == LABEL_TERMS:
                break
    return " ".join(word.title() for word in label)


@dataclass
class _Group:
    total: np.ndarray
    members: list[int] = field(default_factory=list)
    positions: list[int] = field(default_factory=list)
    sentiment: Counter = field(default_factory=Counter)
    topics: set = field(default_factory=set)
    label: str = ""

    @property
    def size(self) -> int:
        return len(self.members)

    def centroid(self) -> np.ndarray:
        mean = self.total / self.size
        norm = np.linalg.norm(mean)
        return mean / norm if norm else mean

    def add(self, row: np.ndarray, index: int, position: int, sentiment: str, topics: set) -> None:
        self.total = self.total + row
        self.members.append(index)
        self.positions.append(position)
        self.sentiment[sentiment] += 1
        self.topics |= topics

    def absorb(self, other: "_Group") -> None:
        self.total = self.total + other.total
        self.members.extend(other.members)
        self.positions.extend(other.positions)
        self.sentiment.update(other.sentiment)
        self.topics |= other.topics

    def dominant_sentiment(self) -> str:
        if not self.sentiment:
            return "neutral"
        label, count = self.sentiment.most_common(1)[0]
        return label if count >= max(2, self.sentiment.total() / 2) else "mixed"


def group_similar(
    texts: list[str],
    sentiments: list[str] | None = None,
    threshold: float = DEFAULT_THRESHOLD,
) -> list[dict]:
    """Cluster comments by meaning.

    Returns one dict per multi-comment group: members, topic label, dominant
    sentiment, and how tightly the members agree (``cohesion``).
    """
    usable = [i for i, text in enumerate(texts) if text and text.strip()]
    documents = build_features([texts[i] for i in usable])
    keep = [i for i, document in enumerate(documents) if document]
    if len(keep) < MIN_GROUP_SIZE:
        return []

    try:
        matrix = TfidfVectorizer(sublinear_tf=True, min_df=1).fit_transform(
            [documents[i] for i in keep]
        )
    except ValueError:
        return []
    if matrix.shape[1] == 0:
        return []

    dense = matrix.toarray()
    sentiments = sentiments or ["neutral"] * len(texts)
    topics = [topic_terms(documents[i].split()) for i in keep]

    groups: list[_Group] = []
    for position, original in enumerate(keep):
        row = dense[position]
        subject = topics[position]
        sentiment = sentiments[original]

        best_index, best_score = -1, 0.0
        for group_index, group in enumerate(groups):
            # Both tests must pass. Cosine alone is fooled by two unrelated
            # comments that merely share an opinion word; the topic overlap is
            # what proves they are discussing the same thing.
            if not (subject & group.topics):
                continue
            score = float(np.dot(group.centroid(), row))
            if score > best_score:
                best_index, best_score = group_index, score

        if best_index >= 0 and best_score >= threshold:
            groups[best_index].add(row, original, position, sentiment, subject)
        else:
            group = _Group(total=row.copy())
            group.add(row, original, position, sentiment, subject)
            groups.append(group)

    # Two groups can open with the same topic words once synonyms are folded
    # ("bad audio" and "audio bad"); merge them so the topic stays singular.
    merged: dict[str, _Group] = {}
    for group in groups:
        group.label = _label_for_group(texts, group.members)
        if group.label in merged:
            merged[group.label].absorb(group)
        else:
            merged[group.label] = group

    results: list[dict] = []
    for group in merged.values():
        if group.size < MIN_GROUP_SIZE:
            continue
        # dense is indexed by position in ``keep``, which is not the same as the
        # original comment index once empty comments have been dropped.
        member_vectors = dense[group.positions]
        pairwise = cosine_similarity(member_vectors)
        size = group.size
        mean_similarity = float(
            (pairwise.sum() - np.trace(pairwise)) / (size * size - size)
        )

        results.append(
            {
                "label": group.label,
                "size": size,
                "share": 0.0,
                "sentiment": group.dominant_sentiment(),
                "sentiment_mix": dict(group.sentiment),
                "cohesion": round(min(1.0, mean_similarity), 3),
                "example": texts[group.members[0]][:160],
                "distinct_phrasings": len({texts[i][:40] for i in group.members}),
                "members": [
                    {"index": i, "text": texts[i][:160]} for i in group.members[:4]
                ],
            }
        )

    total_grouped = sum(row["size"] for row in results)
    for row in results:
        row["share"] = round(row["size"] / total_grouped * 100) if total_grouped else 0

    results.sort(key=lambda row: (-row["size"], -row["cohesion"]))
    return results


def similarity_report(
    texts: list[str],
    sentiments: list[str] | None = None,
    threshold: float = DEFAULT_THRESHOLD,
    top_k: int = 6,
) -> dict:
    """Discussion groups plus the redundancy they expose.

    ``compression`` is the share of comments that add no new discussion, i.e.
    the amount of the raw feed a human no longer has to read.
    """
    total_comments = len([text for text in texts if text and text.strip()])
    if not total_comments:
        return _empty_report()

    groups = group_similar(texts, sentiments, threshold)[:top_k]
    grouped = sum(row["size"] for row in groups)
    redundant = max(0, grouped - len(groups))

    return {
        "comments": total_comments,
        "groups": len(groups),
        "grouped_comments": grouped,
        "redundant": redundant,
        "compression": round(redundant / total_comments * 100, 1),
        "redundant_pairs": sum(row["size"] * (row["size"] - 1) // 2 for row in groups),
        "threshold": threshold,
        "rows": groups,
        "mixed_groups": len([row for row in groups if row["sentiment"] == "mixed"]),
        "redundancy_note": (
            f"{redundant} of {total_comments} comments restate a point already made elsewhere."
            if redundant
            else "No repeated discussion detected in this sample."
        ),
    }


def _empty_report() -> dict:
    return {
        "comments": 0, "groups": 0, "grouped_comments": 0, "redundant": 0,
        "compression": 0.0, "redundant_pairs": 0, "threshold": DEFAULT_THRESHOLD,
        "rows": [], "mixed_groups": 0,
        "redundancy_note": "No comments to compare.",
    }


def duplicate_rate(texts: list[str]) -> float:
    """Share of comments that repeat an earlier comment word for word."""
    seen: set[str] = set()
    duplicates = 0
    total = 0
    for text in texts:
        normalized = " ".join(text.lower().split())
        if not normalized:
            continue
        total += 1
        if normalized in seen:
            duplicates += 1
        else:
            seen.add(normalized)
    return round(duplicates / total * 100, 1) if total else 0.0


__all__ = [
    "AUDITORY_SYNONYMS",
    "DEFAULT_THRESHOLD",
    "EVALUATIVE_SYNONYMS",
    "build_features",
    "duplicate_rate",
    "feature_tokens",
    "group_similar",
    "similarity_report",
]
