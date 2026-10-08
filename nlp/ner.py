"""Experiment 08 - Named Entity Recognition.

Neither spaCy nor an NLTK NER model is installable in this environment
(``maxent_ne_chunker`` was withdrawn from the NLTK index), so this module
implements domain-aware recognition directly. The categories are the ones the
problem statement asks for - people, organizations, locations, products and
technologies - which is a better fit for viewer feedback than the generic
PER / ORG / LOC scheme.

Three recognisers run in order of confidence:

1. **Gazetteer.** Known organizations, technologies, products and places are
   matched as the longest phrase first, so "DaVinci Resolve" wins over
   "Resolve". Single words must be capitalised in the original text, which
   keeps "I love this apple" from becoming the company Apple.
2. **Name patterns.** Honorifics and "Firstname Lastname" pairs built from a
   common-given-name list identify people.
3. **Context fallback.** Any remaining run of capitalised words is an entity
   whose type is inferred from the word beside it: "in Paris" is a place,
   "by Sarah" is a person, "Inc"/"University" is an organization.

Every entity records which recogniser found it, so the dashboard can show that a
"Technology" came from the gazetteer while a "Person" was inferred from context.
"""

from __future__ import annotations

from collections import Counter
from functools import lru_cache

ENTITY_TYPES = ("Organization", "Technology", "Product", "Location", "Person")

#: Phrases matched before anything else, longest first at scan time.
GAZETTEER: dict[str, frozenset[str]] = {
    "Organization": frozenset(
        """
        microsoft google apple amazon meta facebook openai netflix youtube tiktok
        instagram linkedin spotify adobe nvidia intel samsung sony tesla ibm oracle
        salesforce github gitlab reddit twitter stack overflow shopify figma unity
        unreal blender autodesk tesco siemens nokia huawei xiaomi bmw volvo toyota
        honda nasa bbc cnn itv mitsubishi panasonic lg philips sharp fujifilm
        canva notion airbnb uber lyft paypal visa mastercard stripe dropbox
        """.split()
    ),
    "Technology": frozenset(
        """
        python javascript java typescript rust golang ruby php swift kotlin scala
        perl haskell elixir clojure html css sql bash react angular vue svelte
        node django flask fastapi pandas numpy scipy pytorch tensorflow keras
        sklearn nltk spacy transformers linux ubuntu debian windows macos
        android ios chrome firefox safari edge docker kubernetes git github
        jupyter excel vscode intellij figma blender unix apache nginx redis
        postgres mysql mongodb sqlite websocket api json xml regex llm ai ml
        nlp gpu cpu ram ssd hdd csv regex
        mic microphone headphones headset webcam monitor speaker amplifier
        equalizer eq audiointerface soundcard
        """.split()
    ),
    "Product": frozenset(
        """
        chatgpt gemini copilot claude midjourney dalle siri alexa cortana bing
        photoshop illustrator premiere audition after effects lightroom resolve
        handbrake vlc obs davinci premiere pro after effects audacity
        iphone ipad macbook airpod surface kindle playstation xbox nintendo
        whatsapp telegram signal discord zoom slack trello asana notion evernote
        kindle firefox chrome edge
        """.split()
    ),
    "Location": frozenset(
        """
        london paris berlin madrid rome lisbon amsterdam dublin moscow tokyo osaka
        seoul shanghai beijing delhi mumbai bangalore sydney melbourne toronto
        vancouver boston seattle austin denver chicago miami dallas houston
        atlanta portland phoenix detroit minneapolis
        america europe asia africa australia canada china india japan brazil
        mexico france germany italy spain russia korea egypt kenya nigeria
        california texas florida nevada georgia virginia alaska hawaii
        scotland wales ireland england
        """.split()
    ),
}

#: Multi-word phrases, matched case-insensitively because the capitalisation of
#: a product name varies wildly across comments. Each entry is a whole phrase.
PHRASE_ENTITIES: dict[str, tuple[str, ...]] = {
    "Product": (
        "davinci resolve", "obs studio", "after effects", "premiere pro",
        "visual studio code", "windows movie maker", "power point", "notion ai",
    ),
    "Organization": (
        "open ai", "google cloud", "amazon web services", "microsoft office",
        "stanford university", "harvard university", "mit stanford",
    ),
    "Technology": (
        "machine learning", "deep learning", "neural network",
        "large language model", "computer vision", "natural language",
    ),
}

#: Capitalised function words open nearly every comment, so a capitalisation rule
#: alone would report "I" and "The" as entities. These are never names.
ENTITY_STOPWORDS = frozenset(
    """
    the this that these those a an i we they he she it you your my our their his
    her its but and or so if when what why how there here also just very really
    would could should will can do does did is are was were be been being have
    has had not no yes please thanks thank hi hello hey anyone everyone nobody
    everyone something nothing everything one two three first second next last
    new old good great bad love best better sure ok okay yeah oh well now then
    from for to with of into onto about over under again between during against
    within without upon per via while until unless whether how's what's
    """.split()
)

HONORIFICS = frozenset({"mr", "mrs", "ms", "dr", "prof", "professor", "sir", "madam"})

#: Common given names. Kept apart from surnames because "Sarah Chen" is a name
#: while "Mary Anne" is one person with two given names, and only the surname
#: split tells those two cases apart.
FIRST_NAMES = frozenset(
    """
    james john robert michael william david richard joseph thomas charles
    mary patricia jennifer linda elizabeth barbara susan jessica sarah karen
    chris daniel matthew anthony mark donald steven paul andrew joshua kenneth
    kevin brian george timothy ronald edward jason jeffrey ryan jacob gary
    nicole stephanie catherine deborah rebecca sharon laura amy helen anne
    priya arjun ananya rohan sanjay aisha omar fatima yusuf
    hiroshi yuki haruto sara lars ingrid olav sven nils erik bjorn
    olga natasha dmitri svetlana pavel ivan
    """.split()
)

#: Frequent surnames, mainly to avoid mistaking a shared name for a first name.
SURNAMES = frozenset("khan chen wei".split())

GIVEN_NAMES = FIRST_NAMES | SURNAMES

#: Tokens that imply the entity after them is a place or a person. Kept
#: deliberately narrow: "and", "to", "thanks" and "from" appear in almost every
#: comment, so treating them as cues invented entities out of ordinary words.
LOCATION_CUES = frozenset({"near", "around", "outside", "across", "visited"})
PERSON_CUES = frozenset({"by", "with", "follow", "asked", "interviewed"})
ORG_SUFFIXES = frozenset(
    {"inc", "ltd", "llc", "corp", "corporation", "company", "university", "labs", "studio", "group", "foundation"}
)

MAX_PHRASE = 4

#: Brands whose spelling contains internal capitals. Casing is part of the name,
#: so "iphone" in a comment is still the device and not an ordinary word.
MIXED_CASE_BRANDS = frozenset(
    {
        "iPhone", "iPad", "iPod", "macOS", "eBay", "iTunes", "GitHub", "GitLab",
        "YouTube", "JavaScript", "TypeScript", "ChatGPT", "iMessage", "MacBook",
    }
)


@lru_cache(maxsize=1)
def _lookup() -> dict[tuple[str, ...], str]:
    """Normalised phrase -> entity type, for every known entry."""
    table: dict[tuple[str, ...], str] = {}
    for entity_type, phrases in GAZETTEER.items():
        for phrase in phrases:
            if phrase not in ENTITY_STOPWORDS:
                table[tuple(phrase.split())] = entity_type
    for entity_type, phrases in PHRASE_ENTITIES.items():
        for phrase in phrases:
            table[tuple(phrase.split())] = entity_type
    return table


def _is_capitalized(token: str) -> bool:
    return bool(token) and token[0].isupper()


def _classify_by_context(
    previous: str,
    following: str,
    span: list[str],
) -> tuple[str, float] | None:
    """Infer a type for an unknown capitalised span, or reject it.

    Returns ``None`` when the span does not look like a name at all. Comments
    open sentences with capitals constantly, so "Nice", "Export" and "Monday"
    reach this function; without an explicit test for name-likeness every one of
    them became a bogus "Organization" on the dashboard.
    """
    prev = previous.lower()
    last = span[-1].lower()
    lookups = _lookup()

    if last in ORG_SUFFIXES or any(
        word.lower() in ORG_SUFFIXES for word in span[:-1]
    ):
        return "Organization", 0.6

    if following.lower() in {"said", "says", "explained", "wrote", "thinks", "posted", "made"}:
        return "Person", 0.65

    # Reuse a gazetteer type if one word of the span is known.
    for word in span:
        known = lookups.get((word.lower(),))
        if known:
            return known, 0.5

    # Proper names are usually words the lexicon has never seen. A capitalised
    # word that has ordinary everyday meanings is just sentence case.
    if not any(_has_no_common_noun(word) for word in span):
        return None

    if prev in PERSON_CUES:
        return "Person", 0.75
    if prev in LOCATION_CUES:
        return "Location", 0.5
    return "Person", 0.4


@lru_cache(maxsize=8192)
def _has_no_common_noun(word: str) -> bool:
    """True when WordNet knows no everyday (lower-case) sense of the word."""
    from nltk.corpus import wordnet as wn

    try:
        return not wn.synsets(word.lower())
    except Exception:  # noqa: BLE001
        return False


def extract_entities(text: str) -> list[dict]:
    """Find every entity in one comment, with its type and how it was found."""
    from .preprocessor import tokenize_named

    tokens = tokenize_named(text)
    if not tokens:
        return []

    lookup = _lookup()
    lower = [token.lower() for token in tokens]
    entities: list[dict] = []
    consumed = [False] * len(tokens)
    index = 0

    # 1. Gazetteer, longest phrase first.
    while index < len(tokens):
        matched = False
        for size in range(min(MAX_PHRASE, len(tokens) - index), 0, -1):
            span = lower[index : index + size]
            entity_type = lookup.get(tuple(span))
            if entity_type is None and size == 1 and span[0].endswith("s"):
                # Commenters pluralise brand names: GPUs, APIs, iPhones.
                entity_type = lookup.get((span[0][:-1],))
            if entity_type is None:
                continue
            original = tokens[index : index + size]
            # Single words must be capitalised by the commenter, which is what
            # separates the company Apple from the fruit. Brands that are
            # conventionally written in mixed case (iPhone, macOS, GitHub) are
            # exempt, since their casing is a name rather than a sentence start.
            if size == 1 and not (
                _is_capitalized(original[0]) or original[0] in MIXED_CASE_BRANDS
            ):
                continue
            confidence = 0.98 if size > 1 else 0.9
            entities.append(
                {
                    "text": " ".join(original),
                    "type": entity_type,
                    "confidence": confidence,
                    "source": "gazetteer",
                    "start": index,
                    "end": index + size,
                }
            )
            for offset in range(index, index + size):
                consumed[offset] = True
            index += size
            matched = True
            break
        if not matched:
            index += 1

    # 2/3. Remaining runs of capitalised words, resolved by name pattern or context.
    index = 0
    while index < len(tokens):
        if (
            consumed[index]
            or not _is_capitalized(tokens[index])
            or tokens[index].lower() in ENTITY_STOPWORDS
        ):
            index += 1
            continue

        span: list[str] = []
        start = index
        while (
            index < len(tokens)
            and not consumed[index]
            and _is_capitalized(tokens[index])
            and tokens[index].lower() not in ENTITY_STOPWORDS
        ):
            span.append(tokens[index])
            index += 1
        if not span:
            continue
        if len(span) > 3:
            span = span[:3]
            index = start + 3

        previous = tokens[start - 1] if start > 0 else ""
        following = tokens[index] if index < len(tokens) else ""
        lowered = span[0].lower()

        if lowered in HONORIFICS and len(span) > 1:
            entity_type, confidence, source = "Person", 0.85, "honorific"
        elif (
            len(span) >= 2
            and span[0].lower() in FIRST_NAMES
            and span[1].lower() not in FIRST_NAMES
        ):
            entity_type, confidence, source = "Person", 0.8, "name-pattern"
            span = span[:2]
        else:
            verdict = _classify_by_context(previous, following, span)
            if verdict is None:
                # Not a name: skip the span and keep scanning from where the
                # span loop left off, so the cursor still moves forward.
                continue
            entity_type, confidence = verdict
            source = "context"

        entities.append(
            {
                "text": " ".join(span),
                "type": entity_type,
                "confidence": round(confidence, 2),
                "source": source,
                "start": start,
                "end": start + len(span),
            }
        )

    return sorted(entities, key=lambda row: row["start"])


def entity_report(texts: list[str], top_k: int = 8) -> dict:
    """Entity frequency by type, the most-mentioned entities, and examples."""
    if not texts:
        return _empty_report()

    per_type: Counter[str] = Counter()
    mentions: Counter[tuple[str, str]] = Counter()
    sources: Counter[str] = Counter()
    examples: dict[str, list[dict]] = {}

    for text in texts:
        entities = extract_entities(text)
        for entity in entities:
            per_type[entity["type"]] += 1
            mentions[(entity["text"], entity["type"])] += 1
            sources[entity["source"]] += 1
            shown = examples.setdefault(entity["type"], [])
            if len(shown) < 4:
                shown.append(
                    {
                        "text": entity["text"],
                        "sentence": text[:120],
                        "source": entity["source"],
                        "confidence": entity["confidence"],
                    }
                )

    total = sum(per_type.values())
    if not total:
        return _empty_report()

    ranked = sorted(mentions.items(), key=lambda item: (-item[1], item[0][0]))
    top = ranked[:top_k]
    peak = top[0][1] if top else 1

    return {
        "total": total,
        "types": [
            {
                "type": entity_type,
                "count": count,
                "share": round(count / total * 100, 1),
            }
            for entity_type, count in per_type.most_common()
        ],
        "top_entities": [
            {
                "text": name,
                "type": entity_type,
                "count": count,
                "share": round(count / peak * 100),
            }
            for (name, entity_type), count in top
        ],
        "by_source": [
            {"source": source, "count": count}
            for source, count in sources.most_common()
        ],
        "examples": {key: value for key, value in examples.items() if value},
    }


def _empty_report() -> dict:
    return {
        "total": 0,
        "types": [],
        "top_entities": [],
        "by_source": [],
        "examples": {},
    }


__all__ = [
    "ENTITY_TYPES",
    "entity_report",
    "extract_entities",
]
