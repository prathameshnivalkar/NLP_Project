import re
import string

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize

_nltk_ready = False
_tagger_ready: bool | None = None

TAGGER_RESOURCES = ("averaged_perceptron_tagger_eng", "averaged_perceptron_tagger")

LEMMATIZER_POS: dict[str, str] = {
    "NN": "n", "NNS": "n", "NNP": "n", "NNPS": "n",
    "VB": "v", "VBG": "v", "VBD": "v", "VBN": "v", "VBP": "v", "VBZ": "v",
    "JJ": "a", "JJR": "a", "JJS": "a",
    "RB": "r", "RBR": "r", "RBS": "r",
}


def ensure_nltk() -> None:
    global _nltk_ready
    if _nltk_ready:
        return
    for resource in ("punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"):
        try:
            nltk.data.find(f"tokenizers/{resource}" if "punkt" in resource else f"corpora/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)
    _nltk_ready = True


def ensure_tagger() -> bool:
    """Fetch the POS tagger if needed. False when the corpus is unavailable offline."""
    global _tagger_ready
    if _tagger_ready is not None:
        return _tagger_ready

    ensure_nltk()
    for resource in TAGGER_RESOURCES:
        try:
            nltk.data.find(f"taggers/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)
            try:
                nltk.data.find(f"taggers/{resource}")
            except LookupError:
                continue
        _tagger_ready = True
        return True

    _tagger_ready = False
    return False


def tag_tokens(tokens: list[str]) -> list[tuple[str, str]]:
    """Penn Treebank tags, degrading to noun tags when the tagger is missing."""
    if not tokens:
        return []
    try:
        return pos_tag(tokens)
    except LookupError:
        return [(token, "NN") for token in tokens]


_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "]+",
    flags=re.UNICODE,
)
_WHITESPACE_RE = re.compile(r"\s+")
_NON_WORD_RE = re.compile(r"[^A-Za-z'\s]")
_SENTENCE_SPLIT_RE = re.compile(r"[.!?…]+|\n+")

#: Marker inserted where one comment ends a sentence and starts the next.
#: ``_raw_tokens`` only ever emits alphabetic tokens, so this value can never
#: collide with real text, and Experiment 05 refuses to build phrases across it.
SENTENCE_BOUNDARY = "\u2016"

STOP_WORDS: set[str] = set()


def _get_stop_words() -> set[str]:
    global STOP_WORDS
    if not STOP_WORDS:
        ensure_nltk()
        STOP_WORDS = set(stopwords.words("english"))
    return STOP_WORDS


def clean_text(text: str) -> str:
    text = text.lower()
    text = _URL_RE.sub("", text)
    text = _EMOJI_RE.sub("", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def _raw_tokens(text: str) -> list[str]:
    ensure_nltk()
    stop_words = _get_stop_words()
    return [
        token
        for token in word_tokenize(text)
        if token.isalpha() and token not in stop_words and len(token) > 2
    ]


def tokenize_full(text: str) -> list[str]:
    """Every alphabetic token, function words included.

    Experiments 06-08 need this: a statistical tagger reads the words around a
    token, so filtering out "the" and "is" first would strip away the context
    that decides whether a word is a noun or a verb.
    """
    ensure_nltk()
    return [token for token in word_tokenize(clean_text(text)) if token.isalpha()]


def tokenize_surface(text: str) -> list[str]:
    """Inflected surface forms, no lemmatization. Input for Experiment 04."""
    return _raw_tokens(clean_text(text))


def tokenize_surface_by_sentence(text: str) -> list[str]:
    """Surface tokens with :data:`SENTENCE_BOUNDARY` between sentences.

    Experiment 05 needs this so that removing stop-words does not fuse the last
    word of one sentence to the first word of the next.
    """
    tokens: list[str] = []
    for part in _SENTENCE_SPLIT_RE.split(text):
        chunk = _raw_tokens(clean_text(part))
        if not chunk:
            continue
        if tokens:
            tokens.append(SENTENCE_BOUNDARY)
        tokens.extend(chunk)
    return tokens


def tokenize_and_lemmatize(text: str) -> list[str]:
    lemmatizer = WordNetLemmatizer()
    tokens = _raw_tokens(text)
    if not tokens:
        return []
    return [
        lemmatizer.lemmatize(token, LEMMATIZER_POS.get(tag, "n"))
        for token, tag in tag_tokens(tokens)
    ]


def preprocess(text: str) -> str:
    cleaned = clean_text(text)
    tokens = tokenize_and_lemmatize(cleaned)
    return " ".join(tokens)


def preprocess_batch(texts: list[str]) -> list[str]:
    return [preprocess(t) for t in texts]


def tokenize_batch(texts: list[str]) -> list[list[str]]:
    return [tokenize_and_lemmatize(clean_text(t)) for t in texts]


def tokenize_surface_batch(texts: list[str]) -> list[list[str]]:
    return [tokenize_surface(t) for t in texts]


def tokenize_surface_sentences_batch(texts: list[str]) -> list[list[str]]:
    """Sentence-aware surface tokens. Input for Experiment 05."""
    return [tokenize_surface_by_sentence(t) for t in texts]


def tokenize_full_batch(texts: list[str]) -> list[list[str]]:
    """Unfiltered alphabetic tokens. Input for Experiments 06-08."""
    return [tokenize_full(t) for t in texts]


def tokenize_named(text: str) -> list[str]:
    """Alphabetic tokens with their original capitalisation intact.

    Experiment 08 needs this: capitalisation *is* the primary cue for a named
    entity, and :func:`clean_text` lower-cases everything.
    """
    ensure_nltk()
    stripped = _EMOJI_RE.sub(" ", _URL_RE.sub(" ", text))
    stripped = _NON_WORD_RE.sub(" ", stripped)
    return [token for token in word_tokenize(stripped) if token.isalpha()]

