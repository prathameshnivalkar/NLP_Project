"""Experiment 07 - Chunking and Feature Selection.

Two halves of one problem statement.

**Chunking** groups tagged words into noun phrases, verb phrases, adjective
phrases and prepositional phrases. Where Experiment 05 found frequent word
sequences by counting alone, a chunker finds them because of grammar: "the audio
quality" is one thing an author wrote about, not three tokens that happen to sit
next to each other.

**Feature selection** asks which representation actually helps a model. The same
comments are classified four ways - plain words, word + POS, chunks only, and
everything - at four training sizes, and the resulting accuracy table shows
which features earn their place.
"""

from __future__ import annotations

from collections import Counter
from functools import lru_cache
import random

from nltk.chunk import RegexpParser
from nltk.tag import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from .postagger import coarse_tag

#: Rule order matters because the parser is a single greedy left-to-right pass:
#: the noun phrase is matched first so "the audio quality" stays whole, then the
#: adjective phrase takes "very poor" before a verb phrase could absorb it.
#: A bare ``<JJ>`` is deliberately *not* an adjective phrase - an adjective on
#: its own is a modifier, and promoting it would steal words from its noun.
CHUNK_GRAMMAR = r"""
    NP:   {<DT>?<JJ.*>*<NN.*>+}
    ADJP: {<RB>+<JJ.*>|<DT><RB>+<JJ.*>|<JJR>|<JJS>}
    VP:   {<MD>?<VB.*>}
    PP:   {<IN><DT>?<JJ.*>*<NN.*>+}
"""

CHUNK_LABELS: dict[str, str] = {
    "NP": "Noun Phrase",
    "VP": "Verb Phrase",
    "ADJP": "Adjective Phrase",
    "PP": "Prepositional Phrase",
}

#: Tags that carry no topic content once a determiner is stripped.
_FUNCTION_TAGS = {"DT", "CC", "PRP$", "CD"}

#: Representations compared in the feature-selection study.
FEATURE_SETS: dict[str, str] = {
    "unigrams": "Words only",
    "word_pos": "Word + POS",
    "chunks": "Chunks only",
    "all": "Words + POS + chunks",
}

TRAIN_FRACTIONS = (0.25, 0.5, 0.75, 1.0)


@lru_cache(maxsize=1)
def _parser() -> RegexpParser:
    return RegexpParser(CHUNK_GRAMMAR)


@lru_cache(maxsize=4096)
def _tag_cached(tokens: tuple[str, ...]) -> tuple[tuple[str, str], ...]:
    try:
        return tuple(pos_tag(list(tokens)))
    except LookupError:
        from .preprocessor import ensure_tagger

        ensure_tagger()
        return tuple(pos_tag(list(tokens)))


def _sent_tree(tagged: list[tuple[str, str]]):
    return _parser().parse(tagged)


def chunk_sentence(tokens: list[str]) -> list[dict]:
    """Group one tagged sentence into chunks.

    Returns a list of ``{"type", "label", "text", "tokens", "head"}`` dicts.
    Tokens that match no grammar rule are dropped, since a bare adjective with
    no noun and no intensifier is not a phrase an author wrote deliberately.
    """
    if not tokens:
        return []
    tagged = list(_tag_cached(tuple(tokens)))
    chunks: list[dict] = []
    for tree in _sent_tree(tagged):
        if not hasattr(tree, "label") or tree.label() is None:
            continue
        label = tree.label()
        words = [word for word, _ in tree]
        tags = [tag for _, tag in tree]
        if not words:
            continue
        head = next(
            (word for word, tag in zip(reversed(words), reversed(tags)) if tag.startswith("NN")),
            words[-1],
        )
        chunks.append(
            {
                "type": label,
                "label": CHUNK_LABELS.get(label, label),
                "text": " ".join(words),
                "tokens": words,
                "head": head,
                "tags": tags,
            }
        )
    return chunks


def chunk_text(text: str) -> list[dict]:
    from .preprocessor import tokenize_full

    return chunk_sentence(tokenize_full(text))


def _strip_determiner(chunk: dict) -> list[str]:
    return [
        word
        for word, tag in zip(chunk["tokens"], chunk["tags"])
        if tag not in _FUNCTION_TAGS
    ]


def _lemma(word: str) -> str:
    from nltk.stem import WordNetLemmatizer
    from nltk.corpus import wordnet as wn

    try:
        synsets = wn.synsets(word, pos=wn.NOUN)
    except LookupError:
        return word
    return synsets[0].lemmas()[0].name().replace("_", " ") if synsets else word


def key_phrases(
    token_lists: list[list[str]],
    top_k: int = 6,
    min_count: int = 1,
) -> list[dict]:
    """Repeated noun phrases, lemmatised so "tutorials" and "tutorial" merge.

    This is the grammar-driven counterpart to the Experiment 05 n-grams: the
    same underlying discussion, but discovered by phrase structure rather than by
    counting adjacent tokens.
    """
    if not any(token_lists):
        return []

    mentions: Counter[tuple[str, ...]] = Counter()
    documents_seen: Counter[tuple[str, ...]] = Counter()
    total_documents = len([tokens for tokens in token_lists if tokens]) or 1

    for tokens in token_lists:
        if not tokens:
            continue
        in_this_comment: set[tuple[str, ...]] = set()
        for chunk in chunk_sentence(tokens):
            if chunk["type"] not in {"NP", "ADJP"}:
                continue
            words = _strip_determiner(chunk)
            if not words or len(words) > 3:
                continue
            phrase = tuple(_lemma(word) for word in words)
            if len(phrase) == 1 and len(phrase[0]) < 3:
                continue
            mentions[phrase] += 1
            in_this_comment.add(phrase)
        for phrase in in_this_comment:
            documents_seen[phrase] += 1

    if not mentions:
        return []

    ranked = sorted(
        ((phrase, count) for phrase, count in mentions.items() if count >= min_count),
        key=lambda item: (-item[1], -documents_seen[item[0]], item[0]),
    )
    top = ranked[:top_k]
    if not top:
        return []

    peak = top[0][1] or 1
    return [
        {
            "phrase": " ".join(phrase),
            "length": len(phrase),
            "count": count,
            "comments": documents_seen[phrase],
            "share": round(count / peak * 100),
            "comment_share": round(documents_seen[phrase] / total_documents * 100, 1),
        }
        for phrase, count in top
    ]


def chunk_report(token_lists: list[list[str]], top_k: int = 6) -> dict:
    """Chunk type distribution, the phrases they yield, and a worked example."""
    if not any(token_lists):
        return _empty_chunk_report()

    type_counts: Counter[str] = Counter()
    per_sentence: list[int] = []
    example: dict | None = None

    for tokens in token_lists:
        if not tokens:
            continue
        chunks = chunk_sentence(tokens)
        per_sentence.append(len(chunks))
        for chunk in chunks:
            type_counts[chunk["type"]] += 1
        if example is None and len(tokens) > 4:
            example = {
                "text": " ".join(tokens),
                "chunks": [
                    {"label": chunk["label"], "text": chunk["text"]} for chunk in chunks
                ],
            }

    total = sum(type_counts.values()) or 1
    phrases = key_phrases(token_lists, top_k=top_k)

    return {
        "total_chunks": total,
        "avg_per_sentence": round(sum(per_sentence) / len(per_sentence), 2) if per_sentence else 0.0,
        "types": [
            {
                "type": chunk_type,
                "label": CHUNK_LABELS.get(chunk_type, chunk_type),
                "count": count,
                "share": round(count / total * 100, 1),
            }
            for chunk_type, count in type_counts.most_common()
        ],
        "phrases": phrases,
        "example": example,
    }


def _empty_chunk_report() -> dict:
    return {
        "total_chunks": 0,
        "avg_per_sentence": 0.0,
        "types": [],
        "phrases": [],
        "example": None,
    }


# ===== Feature selection and training-data size =====


def _feature_unigrams(tokens: list[str], chunks: list[dict]) -> str:
    return " ".join(tokens)


def _feature_word_pos(tokens: list[str], chunks: list[dict]) -> str:
    return " ".join(f"{word}_{coarse_tag(tag)}" for word, tag in _tag_cached(tuple(tokens)))


def _feature_chunks(tokens: list[str], chunks: list[dict]) -> str:
    return " ".join(chunk["text"] for chunk in chunks) or " ".join(tokens)


def _feature_all(tokens: list[str], chunks: list[dict]) -> str:
    return (
        " ".join(tokens)
        + " || "
        + " ".join(f"{word}_{coarse_tag(tag)}" for word, tag in _tag_cached(tuple(tokens)))
        + " || "
        + " ".join(chunk["text"] for chunk in chunks)
    )


_FEATURE_BUILDERS = {
    "unigrams": _feature_unigrams,
    "word_pos": _feature_word_pos,
    "chunks": _feature_chunks,
    "all": _feature_all,
}


def _score(documents: list[str], labels: list[int], seed: int) -> dict:
    """Train a logistic regression on the smaller split and score the held-out one."""
    unique = set(labels)
    if len(documents) < 10 or len(unique) < 2:
        return {"accuracy": 0.0, "f1": 0.0, "vocabulary": 0}

    x_train, x_test, y_train, y_test = train_test_split(
        documents, labels, test_size=0.3, random_state=seed, stratify=labels
    )
    if len(set(y_train)) < 2 or len(set(y_test)) < 2:
        return {"accuracy": 0.0, "f1": 0.0, "vocabulary": 0}

    vectorizer = TfidfVectorizer(max_features=4000, ngram_range=(1, 2), token_pattern=r"\S+")
    try:
        train_matrix = vectorizer.fit_transform(x_train)
        test_matrix = vectorizer.transform(x_test)
    except ValueError:
        return {"accuracy": 0.0, "f1": 0.0, "vocabulary": 0}

    model = LogisticRegression(max_iter=600, C=4.0, class_weight="balanced")
    model.fit(train_matrix, y_train)
    predicted = model.predict(test_matrix)
    return {
        "accuracy": round(accuracy_score(y_test, predicted) * 100, 1),
        "f1": round(f1_score(y_test, predicted, zero_division=0) * 100, 1),
        "vocabulary": len(vectorizer.vocabulary_),
    }


def feature_selection_study(
    token_lists: list[list[str]],
    labels: list[str],
    seed: int = 42,
) -> dict:
    """Compare feature representations across training-set sizes.

    ``labels`` carries the sentiment already computed by the pipeline, so the
    study measures whether a representation preserves the signal the dashboard
    actually displays, rather than a synthetic label.

    Identical comments are collapsed first. Repeated and near-duplicate comments
    are common in real comment sections, and leaving them in would let the same
    sentence sit in both the training and the test split, which inflates every
    score to 100% and hides the effect being measured.
    """
    seen: set[tuple[str, ...]] = set()
    usable: list[tuple[list[str], str]] = []
    for tokens, label in zip(token_lists, labels):
        if not tokens or label not in {"positive", "negative"}:
            continue
        signature = tuple(tokens)
        if signature in seen:
            continue
        seen.add(signature)
        usable.append((tokens, label))

    if len(usable) < 12:
        return _empty_study(
            f"Needs at least 12 distinct polarised comments; got {len(usable)}."
        )

    token_sets = [tokens for tokens, _ in usable]
    y = [1 if label == "positive" else 0 for _, label in usable]
    cached_chunks = [chunk_sentence(tokens) for tokens in token_sets]

    documents = {
        name: [
            builder(tokens, chunks)
            for (tokens, _), chunks in zip(usable, cached_chunks)
        ]
        for name, builder in _FEATURE_BUILDERS.items()
    }

    # A fixed random permutation keeps every training fraction a random sample
    # of the corpus rather than a prefix, which matters because the comments
    # arrive time-ordered and a prefix would be biased towards one era of the
    # video's life.
    order = list(range(len(y)))
    random.Random(seed).shuffle(order)

    rows: list[dict] = []
    for name, docs in documents.items():
        for fraction in TRAIN_FRACTIONS:
            count = max(6, int(round(len(docs) * fraction)))
            count = min(count, len(docs))
            chosen = order[:count]
            sampled_docs = [docs[i] for i in chosen]
            sampled_y = [y[i] for i in chosen]
            if len(set(sampled_y)) < 2:
                continue
            scores = _score(sampled_docs, sampled_y, seed)
            rows.append(
                {
                    "feature_set": name,
                    "label": FEATURE_SETS[name],
                    "fraction": fraction,
                    "train_size": count,
                    "accuracy": scores["accuracy"],
                    "f1": scores["f1"],
                    "vocabulary": scores["vocabulary"],
                }
            )

    best = max(rows, key=lambda row: (row["f1"], row["accuracy"]), default=None)
    return {
        "documents": len(usable),
        "positive": sum(1 for label in y if label == 1),
        "negative": sum(1 for label in y if label == 0),
        "duplicates_removed": sum(1 for tokens in token_lists if tokens) - len(usable),
        "rows": rows,
        "feature_sets": [
            {"key": key, "label": value} for key, value in FEATURE_SETS.items()
        ],
        "fractions": list(TRAIN_FRACTIONS),
        "best": best,
        "reason": "",
    }


def _empty_study(reason: str = "Not enough polarised comments to train on.") -> dict:
    return {
        "documents": 0,
        "positive": 0,
        "negative": 0,
        "duplicates_removed": 0,
        "rows": [],
        "feature_sets": [
            {"key": key, "label": value} for key, value in FEATURE_SETS.items()
        ],
        "fractions": list(TRAIN_FRACTIONS),
        "best": None,
        "reason": reason,
    }


__all__ = [
    "CHUNK_GRAMMAR",
    "CHUNK_LABELS",
    "FEATURE_SETS",
    "TRAIN_FRACTIONS",
    "chunk_report",
    "chunk_sentence",
    "chunk_text",
    "feature_selection_study",
    "key_phrases",
]
