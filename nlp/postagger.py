"""Experiment 06 - Part-of-Speech Tagging.

Three tagging techniques are run over the same comments so their behaviour can be
compared rather than taken on trust:

``perceptron``
    NLTK's averaged perceptron tagger. Statistical and context sensitive: it
    looks at the words around a token, so ``workflows`` is tagged as a noun in
    "the workflows work" but as a verb in "she workflows her schedule".
``rules``
    A context-free tagger driven by orthographic cues (suffix, capitalisation,
    digits). Fast and predictable, but it has no idea where the word sits.
``hybrid``
    The rule tagger consulted first, with the perceptron as fallback. A lookup
    table is cheaper than a neural pass, so the rules answer first whenever they
    are confident.

The comparison matters for the rest of the pipeline: Experiment 07 chunks and
Experiment 08 entities both consume these tags, so it is worth knowing which
tagger is reliable before building on top of one.
"""

from __future__ import annotations

from collections import Counter
from functools import lru_cache

from nltk import tag as nltk_tag

from .preprocessor import ensure_nltk, ensure_tagger

#: Penn Treebank tags grouped into the coarse categories the dashboard reports.
TAG_GROUPS: dict[str, str] = {
    "NN": "Noun", "NNS": "Noun", "NNP": "Noun", "NNPS": "Noun",
    "VB": "Verb", "VBD": "Verb", "VBG": "Verb", "VBN": "Verb", "VBP": "Verb", "VBZ": "Verb",
    "JJ": "Adjective", "JJR": "Adjective", "JJS": "Adjective",
    "RB": "Adverb", "RBR": "Adverb", "RBS": "Adverb",
    "PRP": "Pronoun", "PRP$": "Pronoun",
    "IN": "Preposition", "DT": "Determiner", "CC": "Conjunction",
    "CD": "Number", "UH": "Interjection", "TO": "Particle", "RP": "Particle",
    "CD.": "Number", "SYM": "Symbol", "FW": "Adverb", "MD": "Verb",
}

#: Orthographic rules, ordered so the first match wins.
RULE_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"^-?[0-9]+([.,][0-9]+)*$", "CD"),
    (r"^#[\w-]+$", "HT"),
    (r"^@[\w-]+$", "NN"),
    (r"^https?$", "URL"),
    (r"(n't)$", "RB"),
    (r"('ll|'re|'ve|'d|'m|'s)$", "MD"),
    (r"(ing)$", "VBG"),
    (r"(edly|ed)$", "VBN"),
    (r"(ies)$", "NNS"),
    (r"(sses|shes|ches|xes|zes)$", "NNS"),
    (r"(s)$", "NNS"),
    (r"(ation|ition|ment|ness|ity|ance|ence|ship|hood|ism|ist)$", "NN"),
    (r"(ful|ous|ive|able|ible|al|ic|ish|less|ary)$", "JJ"),
    (r"(ly)$", "RB"),
    (r"(er|est)$", "JJR"),
    (r"^i$", "PRP"),
)

FUNCTION_WORDS: dict[str, str] = {
    "the": "DT", "a": "DT", "an": "DT", "this": "DT", "that": "DT",
    "these": "DT", "those": "DT", "some": "DT", "any": "DT", "each": "DT",
    "and": "CC", "or": "CC", "but": "CC", "so": "CC", "yet": "CC",
    "is": "VBZ", "am": "VBP", "are": "VBP", "was": "VBD", "were": "VBD",
    "be": "VB", "been": "VBN", "being": "VBG",
    "have": "VBP", "has": "VBZ", "had": "VBD", "having": "VBG",
    "do": "VBP", "does": "VBZ", "did": "VBD", "doing": "VBG",
    "will": "MD", "would": "MD", "can": "MD", "could": "MD", "should": "MD",
    "in": "IN", "on": "IN", "at": "IN", "of": "IN", "for": "IN", "with": "IN",
    "to": "TO", "from": "IN", "by": "IN", "about": "IN", "into": "IN",
    "i": "PRP", "you": "PRP", "he": "PRP", "she": "PRP", "it": "PRP",
    "we": "PRP", "they": "PRP", "my": "PRP$", "your": "PRP$", "our": "PRP$",
    "very": "RB", "really": "RB", "so": "RB", "too": "RB", "also": "RB",
    "not": "RB", "but": "CC", "more": "JJR", "most": "JJS", "best": "JJS",
    "good": "JJ", "great": "JJ", "bad": "JJ", "love": "VB", "hate": "VB",
}

TAGGERS = ("perceptron", "rules", "hybrid")


@lru_cache(maxsize=1)
def _rule_tagger() -> nltk_tag.RegexpTagger:
    ensure_nltk()
    return nltk_tag.RegexpTagger(list(RULE_PATTERNS))


def _perceptron_tag(tokens: list[str]) -> list[tuple[str, str]]:
    ensure_tagger()
    try:
        return nltk_tag.pos_tag(tokens)
    except LookupError:
        return [(token, "NN") for token in tokens]


def _lookup_tag(token: str) -> str | None:
    lowered = token.lower()
    if lowered in FUNCTION_WORDS:
        return FUNCTION_WORDS[lowered]
    if token[:1].isupper() and token[1:].islower():
        return "NNP"
    return None


def _rule_tag_or_none(token: str) -> str | None:
    """Orthographic tag for a single token, or ``None`` when no rule fires."""
    lookup = _lookup_tag(token)
    if lookup:
        return lookup
    tag = _rule_tagger().tag([token])[0][1]
    return None if tag in (None, "None") else tag


def _rules_tag(tokens: list[str]) -> list[tuple[str, str]]:
    return [(token, _rule_tag_or_none(token) or "NN") for token in tokens]


def _hybrid_tag(tokens: list[str]) -> list[tuple[str, str]]:
    """Rules first, perceptron fallback.

    Equivalent to the removed ``nltk.tag.LookaheadTagger(RegexpTagger, ...)``:
    a suffix or capitalisation rule is a cheap, deterministic answer, and the
    statistical tagger only runs where those rules are silent.
    """
    statistical = _perceptron_tag(tokens)
    hybrid: list[tuple[str, str]] = []
    for index, token in enumerate(tokens):
        rule = _rule_tag_or_none(token)
        if rule:
            hybrid.append((token, rule))
        else:
            hybrid.append((token, statistical[index][1] if index < len(statistical) else "NN"))
    return hybrid


_TAG_FUNCTIONS = {
    "perceptron": _perceptron_tag,
    "rules": _rules_tag,
    "hybrid": _hybrid_tag,
}


def tag_with(token_lists: list[list[str]], tagger: str) -> list[list[tuple[str, str]]]:
    """Tag pre-tokenized comments with one named tagger."""
    tag = _TAG_FUNCTIONS.get(tagger)
    if tag is None:
        raise ValueError(f"Unknown tagger '{tagger}'. Choose from {', '.join(TAGGERS)}.")
    return [tag(tokens) for tokens in token_lists if tokens]


def tag_text(text: str, tagger: str = "perceptron") -> list[tuple[str, str]]:
    """Tag a single sentence, keeping the function words preprocessing drops."""
    from .preprocessor import tokenize_full

    tokens = tokenize_full(text)
    if not tokens:
        return []
    return _TAG_FUNCTIONS[tagger](tokens)


def coarse_tag(tag: str) -> str:
    """Collapse a Penn tag into Noun / Verb / Adjective / Adverb and friends."""
    return TAG_GROUPS.get(tag, "Other")


def compare_tagger(token_lists: list[list[str]], tagger: str = "perceptron") -> dict:
    """Distribution, agreement and the biggest disagreements against a baseline.

    A statistical tagger has no ground truth available for arbitrary comments,
    so agreement with the baseline is used as a proxy for stability.
    """
    results = {name: tag_with(token_lists, name) for name in TAGGERS}
    baseline = results.get(tagger) or results["perceptron"]

    distributions: dict[str, dict[str, int]] = {}
    agreements: dict[str, float] = {}
    disagreements: dict[str, list[dict]] = {}
    confusion: Counter[str] = Counter()

    for name, tagged in results.items():
        counts: Counter[str] = Counter()
        for sentence in tagged:
            for _, tag in sentence:
                counts[coarse_tag(tag)] += 1
        total = sum(counts.values()) or 1
        distributions[name] = {
            group: count for group, count in counts.most_common()
        }

        matched = 0
        total_tokens = 0
        examples: list[dict] = []
        for sentence_index, sentence in enumerate(tagged):
            reference = baseline[sentence_index] if sentence_index < len(baseline) else []
            for position, (word, tag) in enumerate(sentence):
                total_tokens += 1
                other = reference[position][1] if position < len(reference) else None
                if other == tag:
                    matched += 1
                else:
                    confusion[f"{coarse_tag(other or '?')} -> {coarse_tag(tag)}"] += 1
                    if len(examples) < 4 and position + 1 < len(sentence):
                        examples.append(
                            {
                                "word": word,
                                "baseline_tag": other or "—",
                                "tag": tag,
                                "context": " ".join(w for w, _ in sentence[max(0, position - 2) : position + 3]),
                            }
                        )
        agreements[name] = round(matched / total_tokens * 100, 1) if total_tokens else 0.0
        disagreements[name] = examples

    return {
        "taggers": list(TAGGERS),
        "baseline": tagger if tagger in results else "perceptron",
        "total_tokens": sum(distributions.get(tagger, {}).values()),
        "distributions": distributions,
        "agreement": agreements,
        "disagreements": disagreements,
        "confusion": [
            {"shift": shift, "count": count}
            for shift, count in confusion.most_common(6)
        ],
    }


def pos_report(token_lists: list[list[str]], top_k: int = 6) -> dict:
    """POS distribution, key content words, and the tagger comparison.

    Experiment 06 asks for a comparison between taggers, not just a single
    tagged output, so the agreement figures are part of the report. The
    distribution itself comes from the perceptron tagger, and
    ``tagger_comparison`` says how far the other two drifted from it.
    """
    if not any(token_lists):
        return _empty_report()

    tagged = tag_with(token_lists, "perceptron")
    comparison = compare_tagger(token_lists)
    counts: Counter[str] = Counter()
    by_group: dict[str, Counter[str]] = {}
    examples: dict[str, list[dict]] = {}

    for sentence in tagged:
        for word, tag in sentence:
            group = coarse_tag(tag)
            counts[group] += 1
            bucket = by_group.setdefault(group, Counter())
            bucket[word.lower()] += 1
            if group in {"Noun", "Adjective", "Verb"}:
                shown = examples.setdefault(group, [])
                if len(shown) < 8 and word.isalpha():
                    shown.append({"word": word, "tag": tag, "group": group})

    total = sum(counts.values()) or 1
    adjectives = [word for word, _ in by_group.get("Adjective", Counter()).most_common(top_k)]
    verbs = [word for word, _ in by_group.get("Verb", Counter()).most_common(top_k)]
    nouns = [word for word, _ in by_group.get("Noun", Counter()).most_common(top_k)]

    return {
        "tokens": total,
        "sentences": len([s for s in tagged if s]),
        "distribution": [
            {"group": group, "count": count, "share": round(count / total * 100, 1)}
            for group, count in counts.most_common()
        ],
        "key_adjectives": adjectives,
        "key_verbs": verbs,
        "key_nouns": nouns,
        "tagger_comparison": [
            {"tagger": name, "agreement": comparison["agreement"].get(name, 0.0)}
            for name in comparison["taggers"]
        ],
        "baseline": comparison["baseline"],
        "examples": {group: rows for group, rows in examples.items() if rows},
    }


def _empty_report() -> dict:
    return {
        "tokens": 0,
        "sentences": 0,
        "distribution": [],
        "key_adjectives": [],
        "key_verbs": [],
        "key_nouns": [],
        "tagger_comparison": [],
        "baseline": "perceptron",
        "examples": {},
    }


__all__ = [
    "TAGGERS",
    "TAG_GROUPS",
    "coarse_tag",
    "compare_tagger",
    "pos_report",
    "tag_text",
    "tag_with",
]
