"""Experiment 04 - Morphological Analysis and Word Generation.

This module decomposes the inflected word forms found in YouTube comments into
their base word plus affixes, and groups every surface form back under a shared
root so keyword counting stops treating ``tutorial`` and ``tutorials`` as two
unrelated terms.

Two jobs live here:

* analysis - a surface form is reduced to its root by taking WordNet's lemma
  first and falling back to an ordered affixation rule table, so out-of-vocabulary
  forms such as ``viewers`` still resolve to ``view``;
* generation - related word forms are produced by applying the affixation rules
  of the word class a word belongs to. Forms come from morphological rules, not
  from a generative model, so nothing here invents new vocabulary.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from functools import lru_cache

from nltk.corpus import wordnet as wn

from .preprocessor import (
    ensure_nltk,
    tag_tokens,
    tokenize_surface_batch,
    tokenize_surface,
)

MIN_ROOT_LENGTH = 2
DOUBLING_PATTERN = "bcdfghjklmnpqrstvwxz"

PENN_TO_LABEL: dict[str, str] = {
    "NN": "Noun", "NNS": "Noun", "NNP": "Noun", "NNPS": "Noun",
    "VB": "Verb", "VBG": "Verb", "VBD": "Verb", "VBN": "Verb",
    "VBP": "Verb", "VBZ": "Verb",
    "JJ": "Adjective", "JJR": "Adjective", "JJS": "Adjective",
    "RB": "Adverb", "RBR": "Adverb", "RBS": "Adverb",
}

# Inflectional suffixes, longest first so "ies" wins over "s". The last element
# is an optional part-of-speech constraint applied only on the rule fallback
# path, where no WordNet lemma was available to disambiguate the form.
INFLECTION_RULES: tuple[tuple[str, str, str | None, str | None], ...] = (
    ("ies", "Plural", "y", None),
    ("sses", "Plural", None, None),
    ("es", "Plural", None, None),
    ("s", "Plural", None, None),
    ("ied", "Past tense", "y", None),
    ("ed", "Past tense", None, None),
    ("ing", "Present participle", None, None),
    ("est", "Superlative", None, "a"),
    ("er", "Comparative", None, "a"),
    ("ly", "Adverbial", None, None),
)

# Derivational suffixes, longest first.
DERIVATION_RULES: tuple[tuple[str, str], ...] = (
    ("iness", "Quality"),
    ("ingly", "Adverbial"),
    ("ation", "Nominalization"),
    ("ment", "Nominalization"),
    ("ness", "Quality"),
    ("ance", "State"),
    ("ence", "State"),
    ("able", "Capability"),
    ("ible", "Capability"),
    ("less", "Quality"),
    ("ful", "Quality"),
    ("ity", "Quality"),
    ("ive", "Tendency"),
    ("ous", "Quality"),
    ("al", "Adjectival"),
    ("ing", "Present participle"),
    ("er", "Agent"),
    ("or", "Agent"),
    ("ly", "Manner"),
)

# Derivational suffixes that force the derived word into a given class.
DERIVATION_POS: dict[str, str] = {
    "ful": "Adjective", "less": "Adjective", "able": "Adjective",
    "ible": "Adjective", "al": "Adjective", "ive": "Adjective",
    "ous": "Adjective", "ly": "Adverb",
}

# Forward affixation rules used by generate_word_forms.
NOUN_FORMS: tuple[tuple[str, str], ...] = (("Plural", "ies"),)
VERB_FORMS: tuple[tuple[str, str], ...] = (
    ("Third person singular", "s"),
    ("Past tense", "ed"),
    ("Present participle", "ing"),
)
ADJ_FORMS: tuple[tuple[str, str], ...] = (
    ("Comparative", "er"),
    ("Superlative", "est"),
)
ADVERB_FORMS: tuple[tuple[str, str], ...] = (("Adverbial", "ly"),)

GENERATION_TEMPLATES: dict[str, tuple[tuple[str, str], ...]] = {
    "Noun": NOUN_FORMS,
    "Verb": VERB_FORMS,
    "Adjective": ADJ_FORMS,
    "Adverb": ADVERB_FORMS,
}


@lru_cache(maxsize=16384)
def _is_known(word: str, pos: str | None) -> bool:
    """Whether WordNet lists this word, used only to pick a sensible label.

    Any NLTK failure is treated as "not known" rather than propagated. This
    function decides a display label, so degrading to the default noun label is
    far better than turning a lookup glitch into a failed request.
    """
    try:
        return bool(wn.synsets(word, pos=[pos] if pos else None))
    except Exception:  # noqa: BLE001
        return False


def _undouble(stem: str) -> str:
    if len(stem) > MIN_ROOT_LENGTH and stem[-1] == stem[-2] and stem[-1] in DOUBLING_PATTERN:
        return stem[:-1]
    return stem


def _restore_silent_e(stem: str) -> str:
    if len(stem) > MIN_ROOT_LENGTH and stem.endswith(("at", "bl", "iz", "us", "iv", "ir")):
        return stem + "e"
    return stem


def _base_candidates(base: str, replace: str | None) -> list[str]:
    """Plausible stems for a stripped form, most likely first.

    Covers consonant doubling (``bigger`` -> ``big``), y-to-i (``happier`` ->
    ``happy``) and silent-e restoration (``amazing`` -> ``amaze``).
    """
    if replace:
        raw = [base + replace]
    else:
        raw = [
            base,
            _undouble(base),
            base + "e",
            base[:-1] + "y" if base.endswith("i") else None,
            _restore_silent_e(base),
            _restore_silent_e(_undouble(base)),
        ]
    candidates: list[str] = []
    for candidate in raw:
        if candidate and len(candidate) >= MIN_ROOT_LENGTH and candidate not in candidates:
            candidates.append(candidate)
    return candidates


_POS_LABEL: dict[str, str] = {"v": "Verb", "n": "Noun", "a": "Adjective", "r": "Adverb"}

# Suffixes are ambiguous across parts of speech (-er, -ed, -ing), so the
# lexicon lookup is ordered by the most likely class for that ending.
_AMBIGUOUS_SUFFIXES: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (("ed", "ing"), ("v", "n", "a", "r")),
    (("er", "est"), ("a", "n", "v", "r")),
    (("ly",), ("r", "a", "n", "v")),
    (("s", "ies"), ("n", "v", "a", "r")),
)


def _ambiguous_order(word: str) -> tuple[str, ...]:
    for suffixes, order in _AMBIGUOUS_SUFFIXES:
        if word.endswith(suffixes):
            return order
    return ("n", "v", "a", "r")


@lru_cache(maxsize=16384)
def _best_lemma(word: str) -> str | None:
    """WordNet's own morphological reduction, or None when the word is already a lemma."""
    for pos in _ambiguous_order(word):
        try:
            result = wn.morphy(word, pos)
        except Exception:  # noqa: BLE001
            return None
        if result and result != word:
            return result
    return None


@lru_cache(maxsize=16384)
def _derivation_root(stem: str) -> tuple[str, list[tuple[str, str]]] | None:
    """Peel lexicon-validated derivational suffixes off a stem."""
    current = stem
    layers: list[tuple[str, str]] = []
    for _ in range(2):
        for suffix, label in DERIVATION_RULES:
            if not current.endswith(suffix) or len(current) - len(suffix) < MIN_ROOT_LENGTH:
                continue
            base = current[: -len(suffix)]
            if not _is_known(base, None):
                continue
            current = base
            layers.append((suffix, label))
            break
        else:
            break
    if not layers:
        return None
    return current, layers


def _match_rule(stem: str, suffix: str, replace: str | None, require_pos: str | None = None) -> str | None:
    """Return a base for a suffix match, or None to reject the rule.

    A base is accepted when the lexicon knows it, or when peeling one
    derivational layer off it yields a known word - that is what lets
    ``viewers`` resolve to ``view`` even though ``viewer`` is not in WordNet.
    """
    if not stem.endswith(suffix) or len(stem) - len(suffix) < MIN_ROOT_LENGTH:
        return None
    for candidate in _base_candidates(stem[: -len(suffix)], replace):
        if _is_known(candidate, require_pos):
            return candidate
        if require_pos is not None:
            continue
        derived = _derivation_root(candidate)
        if derived:
            return candidate
    return None


def decompose(word: str) -> dict:
    """Split a surface form into root plus affixes.

    ``viewers`` -> root ``view``, affixes ``["er", "s"]`` (derivation before
    inflection), ``watching`` -> root ``watch`` with ``["ing"]``.
    """
    ensure_nltk()
    surface = word.lower()
    if not surface.isalpha():
        return {
            "word": word, "root": surface, "lemma": surface, "pos": "Noun",
            "form": "Non-alphabetic", "affixes": [], "affix_details": [],
            "decomposition": surface,
        }

    lemma = _best_lemma(surface)
    stem = lemma or surface
    derivation_layers: list[tuple[str, str]] = []
    forced_pos: str | None = None
    inflection: str | None = None
    inflection_label: str | None = None

    for suffix, label, replace, require_pos in INFLECTION_RULES:
        if not surface.endswith(suffix):
            continue
        if lemma:
            match = lemma if lemma in _base_candidates(surface[: -len(suffix)], replace) else None
        else:
            match = _match_rule(surface, suffix, replace, require_pos)
        if match is not None:
            stem, inflection, inflection_label = match, suffix, label
            break
    else:
        if lemma:
            return _irregular_result(word, lemma)

    derived = _derivation_root(stem)
    if derived:
        stem, derivation_layers = derived
        forced_pos = next(
            (DERIVATION_POS.get(suffix) for suffix, _ in derivation_layers if suffix in DERIVATION_POS),
            None,
        )

    affix_details = [
        {"affix": f"+{suffix}", "kind": "Derivational", "label": label}
        for suffix, label in reversed(derivation_layers)
    ]
    affixes = [detail["affix"].lstrip("+") for detail in affix_details]
    if inflection:
        affixes.append(inflection)
        affix_details.append(
            {"affix": f"+{inflection}", "kind": "Inflectional", "label": inflection_label}
        )

    if inflection_label:
        form = inflection_label
    elif derivation_layers:
        form = "Derived"
    else:
        form = "Base form"

    return {
        "word": word,
        "root": stem,
        "lemma": stem,
        "pos": forced_pos or _guess_pos(stem, form),
        "form": form,
        "affixes": affixes,
        "affix_details": affix_details,
        "decomposition": " + ".join([stem, *affixes]) if affixes else stem,
    }


def _irregular_result(word: str, lemma: str) -> dict:
    return {
        "word": word,
        "root": lemma,
        "lemma": lemma,
        "pos": _guess_pos(lemma, "Base form"),
        "form": "Irregular",
        "affixes": [],
        "affix_details": [],
        "decomposition": f"{lemma} → {word}",
    }


_POS_PREFERENCE: dict[str, tuple[str, ...]] = {
    "Plural": ("n", "a", "v", "r"),
    "Comparative": ("a", "n", "v", "r"),
    "Superlative": ("a", "n", "v", "r"),
    "Past tense": ("v", "n", "a", "r"),
    "Present participle": ("v", "n", "a", "r"),
    "Adverbial": ("r", "a", "n", "v"),
}


def _guess_pos(root: str, form: str) -> str:
    for pos in _POS_PREFERENCE.get(form, ("n", "v", "a", "r")):
        if _is_known(root, pos):
            return _POS_LABEL[pos]
    return "Noun"


@lru_cache(maxsize=16384)
def _cached_decompose(word: str) -> dict:
    return decompose(word)


def analyze_word(word: str, pos_label: str | None = None) -> dict:
    """Morphological analysis of one word, with a POS tag when one is supplied."""
    analysis = dict(_cached_decompose(word))
    if pos_label and pos_label in PENN_TO_LABEL.values():
        analysis["pos"] = pos_label
    return analysis


def analyze_text(text: str) -> list[dict]:
    """Tokenize a comment and morphologically analyse every surface form."""
    tokens = tokenize_surface(text)
    if not tokens:
        return []
    tagged = tag_tokens(tokens)
    return [
        {**analyze_word(token, PENN_TO_LABEL.get(tag)), "tag": tag}
        for token, tag in tagged
    ]


VOWELS = "aeiouy"

POS_CODES: dict[str, str] = {"Noun": "n", "Verb": "v", "Adjective": "a", "Adverb": "r"}


def _syllables(word: str) -> int:
    count = 0
    previous_was_vowel = False
    for char in word:
        is_vowel = char in VOWELS
        if is_vowel and not previous_was_vowel:
            count += 1
        previous_was_vowel = is_vowel
    return max(count, 1)


def _apply_affix(base: str, suffix: str) -> str:
    if suffix == "ies":
        if base.endswith(("s", "x", "z", "ch", "sh")):
            return base + "es"
        if base.endswith("y") and len(base) > 2 and base[-2] not in "aeiou":
            return base[:-1] + "ies"
        return base + "s"

    if suffix not in {"s", "es", "ed", "ing", "er", "est", "ly"}:
        return base + suffix

    if base.endswith("e") and suffix in {"ed", "ing", "er", "est"}:
        return base[:-1] + suffix

    # A consonant + y base keeps the y before -ing (study -> studying) but
    # turns it into -ies for the plural (study -> studies) and -i elsewhere
    # (happy -> happier, happily).
    if base.endswith("y") and len(base) > 2 and base[-2] not in "aeiou":
        if suffix in {"s", "es"}:
            return base[:-1] + "ies"
        if suffix == "ing":
            return base + "ing"
        return base[:-1] + "i" + suffix

    if base.endswith(("s", "x", "z", "ch", "sh")) and suffix in {"s", "es"}:
        return base + ("es" if suffix == "s" else suffix)

    # Final-consonant doubling applies to the tensed forms of monosyllables
    # (stop -> stopped, run -> running) and never to the third person
    # singular or to polysyllabic words like "edited".
    if (
        suffix in {"ed", "ing"}
        and len(base) > 2
        and base[-1] not in "aeiouwxy"
        and base[-2] in "aeiou"
        and base[-3] not in "aeiou"
        and _syllables(base) == 1
    ):
        return base + base[-1] + suffix

    return base + suffix


@lru_cache(maxsize=1)
def _lemma_names() -> frozenset[str]:
    """Every lemma name WordNet indexes, used to attest generated word forms."""
    ensure_nltk()
    try:
        return frozenset(name.lower() for name in wn.all_lemma_names())
    except Exception:  # noqa: BLE001
        return frozenset()


def _is_attested(word: str) -> bool:
    """True when WordNet indexes ``word`` as a lemma of its own.

    Unlike ``_is_known`` this does not accept a form that merely morphs to a
    known word, so ``tutorialer`` is rejected while ``quickest`` is accepted.
    """
    names = _lemma_names()
    return word in names if names else _is_known(word, None)


def _generate_for_class(base: str, word_class: str) -> list[dict]:
    forms = [{"form": base, "label": "Base form", "rule": "", "word_class": word_class}]
    for form_label, suffix in GENERATION_TEMPLATES[word_class]:
        candidate = _apply_affix(base, suffix)
        if candidate == base or any(f["form"] == candidate for f in forms):
            continue
        forms.append(
            {
                "form": candidate,
                "label": form_label,
                "rule": f"+{suffix}",
                "word_class": word_class,
            }
        )
    return forms


def generate_word_forms(word: str, pos_label: str | None = None) -> list[dict]:
    """Rule-based word-form generation for a base word.

    ``watch`` yields ``watches``, ``watched`` and ``watching`` and ``tutorial``
    yields ``tutorials``, by applying the affixation rules of the word class the
    morphological analysis settled on. Where a word is ambiguous across classes
    (``watch`` is both noun and verb) the class that yields the most attested
    forms wins, with the analysed class breaking ties. Forms come from
    linguistic rules - nothing here invents new vocabulary.
    """
    analysis = analyze_word(word, pos_label)
    base = analysis["root"]
    preferred = analysis["pos"]

    order = [preferred] + [name for name in POS_CODES if name != preferred]
    best_forms: list[dict] = []
    best_score = -1.0
    for index, word_class in enumerate(order):
        forms = _generate_for_class(base, word_class)
        attested = sum(1 for f in forms[1:] if _is_attested(f["form"]))
        # One attested form must outweigh a whole class of the ranked order, so
        # "quick" resolves to its adjective forms while "tutorial" stays a noun.
        score = attested - index * 0.5
        if score > best_score:
            best_forms, best_score = forms, score

    best_forms[0]["word_class"] = preferred
    return best_forms


def normalize_tokens(tokens: list[str]) -> list[str]:
    """Collapse inflected surface forms onto shared roots (Experiment 04 -> keywords)."""
    return [_cached_decompose(token)["root"] for token in tokens]


def normalize_batch(token_lists: list[list[str]]) -> list[list[str]]:
    return [normalize_tokens(tokens) for tokens in token_lists]


def morphology_report(texts: list[str], top_n: int = 8) -> dict:
    """Corpus-level morphological analysis used by the dashboard."""
    if not texts:
        return _empty_report()

    token_lists = tokenize_surface_batch(texts)
    if not any(token_lists):
        return _empty_report()

    form_counts: Counter[str] = Counter()
    pos_counts: Counter[str] = Counter()
    affix_counts: Counter[tuple[str, str, str]] = Counter()
    root_forms: dict[str, set[str]] = defaultdict(set)
    root_pos: dict[str, Counter[str]] = defaultdict(Counter)
    root_counts: Counter[str] = Counter()
    analyses: dict[str, dict] = {}

    for tokens in token_lists:
        if not tokens:
            continue
        for token, tag in tag_tokens(tokens):
            analysis = _cached_decompose(token)
            analyses[token] = analysis
            form_counts[token] += 1
            pos_counts[analysis["pos"]] += 1
            root_counts[analysis["root"]] += 1
            root_forms[analysis["root"]].add(token)
            root_pos[analysis["root"]][analysis["pos"]] += 1
            for detail in analysis["affix_details"]:
                affix_counts[(detail["affix"], detail["kind"], detail["label"])] += 1

    total_tokens = sum(form_counts.values())
    if not total_tokens:
        return _empty_report()

    changed = sum(count for word, count in form_counts.items() if word != analyses[word]["root"])
    collapse_ratio = round((1 - len(root_counts) / len(form_counts)) * 100, 1) if form_counts else 0.0

    top_forms = [
        {
            "word": word,
            "root": analyses[word]["root"],
            "pos": analyses[word]["pos"],
            "form": analyses[word]["form"],
            "affixes": analyses[word]["affixes"],
            "decomposition": analyses[word]["decomposition"],
            "count": count,
            "share": round(count / total_tokens * 100, 1),
        }
        for word, count in form_counts.most_common(top_n)
    ]

    top_roots = [
        {
            "root": root,
            "pos": root_pos[root].most_common(1)[0][0],
            "forms": sorted(root_forms[root]),
            "count": count,
            "generated": [f["form"] for f in generate_word_forms(root)],
        }
        for root, count in root_counts.most_common(top_n)
    ]

    affix_breakdown = [
        {
            "affix": affix,
            "kind": kind,
            "label": label,
            "count": count,
            "share": round(count / total_tokens * 100, 1),
        }
        for (affix, kind, label), count in affix_counts.most_common(6)
    ]

    pos_breakdown = [
        {"pos": pos, "count": count, "share": round(count / total_tokens * 100, 1)}
        for pos, count in pos_counts.most_common()
    ]

    return {
        "tokens_analyzed": total_tokens,
        "distinct_forms": len(form_counts),
        "distinct_roots": len(root_counts),
        "inflected_tokens": changed,
        "collapse_ratio": collapse_ratio,
        "forms": top_forms,
        "root_clusters": top_roots,
        "affix_breakdown": affix_breakdown,
        "pos_breakdown": pos_breakdown,
    }


def _empty_report() -> dict:
    return {
        "tokens_analyzed": 0,
        "distinct_forms": 0,
        "distinct_roots": 0,
        "inflected_tokens": 0,
        "collapse_ratio": 0.0,
        "forms": [],
        "root_clusters": [],
        "affix_breakdown": [],
        "pos_breakdown": [],
    }


__all__ = [
    "analyze_text",
    "analyze_word",
    "decompose",
    "ensure_nltk",
    "generate_word_forms",
    "morphology_report",
    "normalize_batch",
    "normalize_tokens",
]
