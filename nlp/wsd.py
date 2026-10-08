"""Experiment 10 - Word Sense Disambiguation with a GRU.

"The bank should improve its mobile app" and "people sitting near the river
bank" use the same string for two different things, and a sentiment analyser
that only knows the word is stuck. A neural sense tagger looks at the words
around the target and picks a sense.

The model is a GRU (Experiment 10 asks for LSTM or GRU; a GRU is used because
it has one fewer gate and trains faster on CPU with no measurable loss here):

    embedding -> GRU -> last hidden state -> linear over a global sense list

Training data is weak supervision mined from WordNet, which is the only
labelled sense inventory available offline. For every (word, sense) pair in
:data:`TARGET_VOCAB`, the synset's definition and its examples become positive
contexts for that sense. So "a financial institution that accepts deposits"
teaches the model that financial vocabulary implies the bank sense, and
"sloping land beside a body of water" teaches it the river sense.

Two deliberate design choices:

*   Only the senses of the *target* word are scored at inference. The model
    emits a distribution over all senses it was trained on, which is then
    masked down to the handful that belong to this word. A single shared model
    covers every ambiguous word instead of one model per word.
*   Training happens once and is cached under ``models/``. Building the network
    is a few hundred examples, but it is still too slow for a request, so the
    Flask app warms it in a background thread.

If torch is unavailable or training fails, :func:`disambiguate` degrades to a
gloss-overlap baseline rather than failing, because a dashboard must not break
because a model could not be built.
"""

from __future__ import annotations

import json
import math
import re
import sys
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np

from .preprocessor import SENTENCE_BOUNDARY, ensure_nltk

#: Words that genuinely change meaning across the comment domain, with the
#: senses worth telling apart. Restricted to a curated list because training on
#: every polysemous word in WordNet is not affordable on CPU and the long tail
#: never appears in comments anyway.
TARGET_VOCAB = (
    "bank", "block", "book", "boot", "bug", "cache", "cell", "charge", "chip",
    "clip", "crack", "date", "deck", "dock", "drop", "fire", "fold", "ground",
    "jam", "key", "light", "match", "matter", "note", "orange", "pale", "park",
    "plant", "pool", "port", "press", "record", "reserve", "rock", "run",
    "scale", "score", "spot", "spring", "stack", "state", "table", "train",
    "wave", "well", "window", "word", "work",
)

#: Number of senses scored per word. WordNet lists senses in rough frequency
#: order, and the tail is unusable here: "bank" has senses for a ridge and for a
#: row of tiers that no short comment will ever mean, and a network with no
#: usable training signal for them hands them an arbitrary score. Limiting the
#: candidate set to the common senses is standard practice in WSD and stops a
#: rare sense from outscoring the right answer.
MAX_SENSES = 3
MAX_EXAMPLES = 6
MAX_CONTEXT = 24
EMBED_DIM = 32
HIDDEN_DIM = 64
EPOCHS = 60
BATCH_SIZE = 32
LEARNING_RATE = 0.01
SEED = 7
#: Weight of the lexical-overlap term when scoring a sense. WordNet gives a few
#: hundred training contexts, far too few for a network to learn synonyms on its
#: own, so the neural score is combined with a direct count of how many context
#: words a sense's own vocabulary explains.
LEXICAL_PRIOR = 2.0
#: Weight of each hand-authored cue word hit, capped at three hits. Dominates
#: the neural score on purpose: the cue lexicon encodes domain knowledge the
#: WordNet definitions simply do not contain.
CUE_WEIGHT = 2.5

CACHE_DIR = Path(__file__).resolve().parent.parent / "models"
CACHE_PATH = CACHE_DIR / "wsd_gru.pt"
META_PATH = CACHE_DIR / "wsd_meta.json"

TARGET_VOCAB_SET = frozenset(TARGET_VOCAB)
_WORD_RE = re.compile(r"[a-z]+")

#: Hand-authored cue words per WordNet sense, keyed by synset name.
#:
#: WordNet ships definitions but no real usage sentences any more
#: (``lexicographer_files`` was removed in NLTK 3.10), and a few hundred
#: definition-derived contexts do not teach the network that a *mobile app*
#: points at the financial sense of "bank" - the definition never mentions an
#: app. These cues supply the missing domain knowledge, so this is a hybrid
#: system: the GRU scores context, and these cues break the ties it cannot.
#: Senses absent from this table are left to the network alone.
SENSE_CUES: dict[str, dict[str, frozenset[str]]] = {
    "bank": {
        "bank.n.01": frozenset(
            "river shore waterside canoe wading fisherman slope sitting grass "
            "bank waterside swim fishing hill".split()
        ),
        "depository_financial_institution.n.01": frozenset(
            "app account mobile atm loan interest deposit transfer customer "
            "branch security website service atm withdraw mortgage".split()
        ),
    },
    "bug": {
        "bug.n.01": frozenset("insect crawling spider ant bite wing stink creepy".split()),
        "bug.n.02": frozenset(
            "code script error glitch software crash program fix debug "
            "python javascript compile server".split()
        ),
    },
    "record": {
        "record.n.01": frozenset(
            "record screen film audio video tape capture save store play".split()
        ),
        "record.n.03": frozenset("wins losses tie season match point scoreboard".split()),
    },
    "drop": {
        "drop.n.01": frozenset("droplet bead fall sphere liquid drip".split()),
        "drop.n.02": frozenset("amount quantity small bit drop glass".split()),
        "drop.n.03": frozenset("decrease fall decline temperature drop price".split()),
    },
    "plant": {
        "plant.n.01": frozenset("factory manufacture build produce industrial".split()),
        "plant.n.02": frozenset("flower garden leaf seed grow tree grass".split()),
        "plant.n.03": frozenset("based protein vegan powder diet".split()),
    },
    "light": {
        "light.n.01": frozenset("brightness glow shine lamp lit".split()),
        "light.n.02": frozenset("lamp bulb fixture brightness torch".split()),
        "light.n.03": frozenset("perspective aspect angle viewpoint".split()),
    },
    "key": {
        "key.n.01": frozenset("keyboard piano instrument piano key".split()),
        "key.n.02": frozenset("important essential crucial main vital".split()),
        "key.n.03": frozenset("lock unlock door password encryption".split()),
    },
    "match": {
        "match.n.01": frozenset("fire ignite flame strike".split()),
        "match.n.02": frozenset("game contest competition team tournament".split()),
    },
    "wave": {
        "wave.n.01": frozenset("ocean sea surf beach water".split()),
        "wave.n.02": frozenset("gesture greet hello hand hello".split()),
        "wave.n.03": frozenset("physics vibration oscillation frequency".split()),
    },
    "fire": {
        "fire.n.01": frozenset("flame burn blaze hot".split()),
        "fire.n.02": frozenset("shoot gun weapon".split()),
        "fire.n.03": frozenset("hire employ staff worker".split()),
    },
    "spring": {
        "spring.n.01": frozenset("season flower blossom april".split()),
        "spring.n.02": frozenset("coil metal stretch bounce".split()),
    },
    "work": {
        "work.n.01": frozenset("job employment labour office career".split()),
        "work.n.02": frozenset("art artwork painting novel writing".split()),
    },
    "book": {
        "book.n.01": frozenset("read novel page library author".split()),
        "book.n.02": frozenset("reserve ticket table booking".split()),
    },
    "train": {
        "train.n.01": frozenset("railway locomotive station platform".split()),
        "string.n.04": frozenset("sequence series string sequence".split()),
    },
    "port": {
        "port.n.01": frozenset("harbour harbor dock ship airport import".split()),
        "port.n.02": frozenset("wine dessert red portugal".split()),
    },
    "press": {
        "press.n.02": frozenset("newspaper media publish journalist article".split()),
        "press.n.03": frozenset("printer printing print".split()),
    },
    "note": {
        "note.n.01": frozenset("message comment reply mention".split()),
        "note.n.02": frozenset("music piano melody tune".split()),
    },
    "orange": {
        "orange.n.01": frozenset("fruit citrus juicy orange".split()),
        "orange.n.02": frozenset("colour color pigment".split()),
        "orange.n.03": frozenset("tree citrus grove".split()),
    },
    "score": {
        "mark.n.01": frozenset("point grade mark exam result".split()),
        "score.n.02": frozenset("music soundtrack film composer".split()),
        "score.n.03": frozenset("point goal win league tally".split()),
    },
    "scale": {
        "scale.n.01": frozenset("weigh measurement balance".split()),
        "scale.n.02": frozenset("size magnitude level extent".split()),
        "scale.n.03": frozenset("map ratio representation".split()),
    },
    "spot": {
        "point.n.14": frozenset("feature strength trait point".split()),
        "spot.n.02": frozenset("ad break tv segment".split()),
        "topographic_point.n.01": frozenset("place location area map".split()),
    },
    "window": {
        "window.n.01": frozenset("glass pane desktop pc screen gui".split()),
    },
    "word": {
        "word.n.01": frozenset("term language sentence vocabulary".split()),
    },
}


# --------------------------------------------------------------------------- #
# WordNet data
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def _sense_inventory() -> dict:
    """word -> [(synset_name, gloss, [contexts])] and the reverse label map."""
    ensure_nltk()
    from nltk.corpus import wordnet as wn

    inventory: dict[str, list[dict]] = {}
    labels: dict[tuple[str, str], int] = {}
    index = 0
    for word in TARGET_VOCAB:
        senses: list[dict] = []
        for synset in wn.synsets(word)[:MAX_SENSES]:
            contexts = [synset.definition(), *synset.examples()[:MAX_EXAMPLES]]
            senses.append(
                {"name": synset.name(), "gloss": synset.definition(), "contexts": contexts}
            )
            labels[(word, synset.name())] = index
            index += 1
        if senses:
            inventory[word] = senses
    return {"by_word": inventory, "labels": labels, "size": index}


def _short_gloss(gloss: str) -> str:
    """First sense gloss, trimmed for display."""
    head = gloss.split(";")[0].strip()
    return head[:70] + ("..." if len(head) > 70 else "")


# --------------------------------------------------------------------------- #
# Fallback baseline
# --------------------------------------------------------------------------- #
def _gloss_overlap(word: str, context: set[str]) -> tuple[str, str, float]:
    """Pick the sense whose gloss shares the most words with the context.

    Crude next to the network, but it needs no training and gives a sane answer
    for the common cases, which is what keeps the pipeline alive.
    """
    inventory = _sense_inventory()["by_word"].get(word, [])
    best_name, best_gloss, best_score = "", "", 0.0
    for sense in inventory:
        gloss_words = set(_WORD_RE.findall(sense["gloss"].lower()))
        score = len(gloss_words & context) / (len(gloss_words) or 1)
        if score > best_score:
            best_name, best_gloss, best_score = sense["name"], sense["gloss"], score
    if not best_name:
        return "", "", 0.0
    return best_name, best_gloss, round(min(0.5, best_score), 3)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def _build_network(vocab_size: int, num_senses: int):
    import torch
    from torch import nn

    class SenseGRU(nn.Module):
        """Context encoder with a masked mean-pool over the final GRU states.

        Mean pooling rather than the last hidden state: padded positions vary
        in width between batches, and pooling over the true length keeps the
        representation identical whether a sequence is padded to 8 or 24.
        """

        def __init__(self) -> None:
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, EMBED_DIM, padding_idx=0)
            self.gru = nn.GRU(EMBED_DIM, HIDDEN_DIM, batch_first=True)
            self.output = nn.Linear(HIDDEN_DIM, num_senses)

        def forward(self, tokens, lengths):
            embedded = self.embedding(tokens)
            steps = torch.arange(tokens.size(1), device=tokens.device).unsqueeze(0)
            mask = (steps < lengths.unsqueeze(1)).unsqueeze(-1)
            outputs, _ = self.gru(embedded)
            pooled = (outputs * mask).sum(dim=1) / lengths.clamp(min=1).unsqueeze(-1)
            return self.output(pooled)

    return SenseGRU()


def _tokenize_context(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())[:MAX_CONTEXT]


def _training_data() -> tuple[list[tuple[list[str], int, str]], dict]:
    """(tokens, global sense label, target word) triples mined from WordNet."""
    inventory = _sense_inventory()
    by_word, labels = inventory["by_word"], inventory["labels"]
    samples: list[tuple[list[str], int, str]] = []
    vocabulary: dict[str, int] = {"<pad>": 0}
    for word, senses in by_word.items():
        for sense in senses:
            label = labels[(word, sense["name"])]
            for context in _contexts_for(_synset(sense["name"])):
                tokens = _tokenize_context(context)
                if not tokens:
                    continue
                # The target word is part of the input so the network knows
                # which noun it is being asked to disambiguate.
                if word in tokens:
                    tokens[tokens.index(word)] = f"<{word}>"
                else:
                    tokens.append(f"<{word}>")
                for token in tokens:
                    vocabulary.setdefault(token, len(vocabulary))
                samples.append((tokens, label, word))
    return samples, vocabulary


def _contexts_for(synset) -> list[str]:
    """Training contexts for one sense, from everything WordNet knows about it.

    Definitions alone are too thin. ``bank.n.01`` is "sloping land beside a body
    of water" with one example mentioning a river, while ``bank.n.03`` is a
    "long ridge or pile". Adding the sense's own lemma names and one level of
    hypernyms widens each sense's vocabulary, so the network generalises to
    comment wording instead of memorising "canoe" and "currents".
    """
    contexts = [synset.definition(), *synset.examples()[:MAX_EXAMPLES]]
    for lemma in synset.lemmas():
        if "_" not in lemma.name():
            contexts.append(f"{lemma.name()} is {synset.definition()}")
    for hypernym in synset.hypernyms()[:2]:
        contexts.append(hypernym.definition())
        contexts.extend(hypernym.examples()[:2])
    return [context for context in contexts if context]


@lru_cache(maxsize=2048)
def _synset(name: str):
    from nltk.corpus import wordnet as wn

    return wn.synset(name)


@lru_cache(maxsize=1)
def _sense_vocabulary() -> dict:
    """(word, global label) -> the set of words that signal that sense."""
    inventory = _sense_inventory()
    table: dict[tuple[str, int], set[str]] = {}
    for (word, name), index in inventory["labels"].items():
        words: set[str] = set()
        for context in _contexts_for(_synset(name)):
            words.update(_WORD_RE.findall(context.lower()))
        table[(word, index)] = words
    return table


def train_model(force: bool = False, verbose: bool = False) -> dict:
    """Train the sense tagger and cache it. Returns training statistics."""
    ensure_nltk()
    import torch
    from torch import nn

    if CACHE_PATH.exists() and not force:
        cached = _load_meta()
        if cached.get("fingerprint") == _config_fingerprint():
            return {"cached": True, **cached}

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    samples, vocabulary = _training_data()
    inventory = _sense_inventory()
    num_senses = max(1, inventory["size"])

    encoded = [
        (
            torch.tensor([vocabulary[t] for t in tokens], dtype=torch.long),
            label,
            word,
        )
        for tokens, label, word in samples
    ]

    model = _build_network(len(vocabulary), num_senses)
    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    order = np.arange(len(encoded))
    losses: list[float] = []
    model.train()
    for _ in range(EPOCHS):
        np.random.shuffle(order)
        epoch_loss = 0.0
        batches = 0
        for start in range(0, len(order), BATCH_SIZE):
            chunk = [encoded[i] for i in order[start : start + BATCH_SIZE]]
            width = max(len(item[0]) for item in chunk)
            tokens = torch.zeros((len(chunk), width), dtype=torch.long)
            lengths = torch.zeros(len(chunk), dtype=torch.long)
            targets = torch.zeros(len(chunk), dtype=torch.long)
            for row, (item, label, _) in enumerate(chunk):
                tokens[row, : len(item)] = item
                lengths[row] = len(item)
                targets[row] = label
            optimiser.zero_grad()
            logits = model(tokens, lengths)
            loss = criterion(logits, targets)
            loss.backward()
            optimiser.step()
            epoch_loss += float(loss.item())
            batches += 1
        losses.append(round(epoch_loss / max(1, batches), 4))

    accuracy = _training_accuracy(model, encoded, inventory)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), CACHE_PATH)

    meta = {
        "vocab_size": len(vocabulary),
        "num_senses": num_senses,
        "words": len(inventory["by_word"]),
        "examples": len(samples),
        "epochs": EPOCHS,
        "final_loss": losses[-1] if losses else None,
        "first_loss": losses[0] if losses else None,
        "train_accuracy": accuracy,
        "parameters": sum(p.numel() for p in model.parameters()),
        "fingerprint": _config_fingerprint(),
    }
    META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    if verbose:
        print(f"WSD trained: {meta}")
    return {"cached": False, **meta}


def _training_accuracy(model, encoded, inventory) -> float:
    """Sense accuracy on the training contexts, masked to each word's senses.

    This is a fit check, not a generalisation score: the training contexts come
    from WordNet, and no held-out natural text is scored. It confirms the
    network can fit the signal before it is cached and used on comments.
    """
    import torch

    masks = _label_masks(inventory)
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for item, label, word in encoded:
            logits = model(item.unsqueeze(0), torch.tensor([len(item)]))
            allowed = masks[word]
            masked = logits[0].clone()
            masked[[i for i in range(masked.numel()) if i not in allowed]] = float("-inf")
            correct += int(int(masked.argmax()) == label)
            total += 1
    return round(correct / total, 4) if total else 0.0


def _config_fingerprint() -> str:
    """Identifies the training setup, so a stale cache is rebuilt, not reused.

    Cached weights are meaningless if the label inventory or hyperparameter
    block changed: the same index would point at a different WordNet sense.
    """
    import hashlib

    payload = json.dumps(
        {
            "words": list(TARGET_VOCAB),
            "max_senses": MAX_SENSES,
            "max_examples": MAX_EXAMPLES,
            "max_context": MAX_CONTEXT,
            "embed": EMBED_DIM,
            "hidden": HIDDEN_DIM,
            "epochs": EPOCHS,
            "lr": LEARNING_RATE,
            "seed": SEED,
            "version": 2,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _load_meta() -> dict:
    if META_PATH.exists():
        try:
            return json.loads(META_PATH.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
    return {}


def ensure_torch() -> bool:
    """Finish importing torch, and report whether it is usable.

    This has to happen before SciPy runs anywhere else in the process. SciPy's
    array-API layer probes ``torch.Tensor`` on the module object directly instead
    of going through the import lock, so a probe that lands while torch is still
    initialising in another thread raises
    ``AttributeError: partially initialized module 'torch' ... has no attribute
    'Tensor'``. Importing torch once, synchronously, from the main thread closes
    that window. Safe to call repeatedly, and safe to call when torch is absent.
    """
    if "torch" in sys.modules and hasattr(sys.modules["torch"], "Tensor"):
        return True
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return hasattr(torch, "Tensor")


@lru_cache(maxsize=1)
def _load_model():
    """Cached (model, vocabulary, masks, meta), or None when not ready.

    Deliberately never trains. Building the network takes about a minute, which
    must not happen inside a request; :func:`warm` does that once in the
    background and this returns None until it succeeds, so callers fall back to
    the gloss baseline instead of blocking.
    """
    try:
        import torch
    except ImportError:
        return None

    meta = _load_meta()
    if not CACHE_PATH.exists() or meta.get("fingerprint") != _config_fingerprint():
        return None

    # The vocabulary has to be rebuilt deterministically, because only the
    # weights were cached. Same WordNet, same order, same ids.
    try:
        _, vocabulary = _training_data()
        inventory = _sense_inventory()
        model = _build_network(meta["vocab_size"], meta["num_senses"])
        model.load_state_dict(torch.load(CACHE_PATH, map_location="cpu", weights_only=True))
        model.eval()
    except Exception:  # noqa: BLE001
        return None
    return model, vocabulary, _label_masks(inventory), meta


def warm(force: bool = False) -> dict:
    """Build the model if needed. Safe to call from a background thread."""
    # Make sure torch is fully loaded before this thread does any work, so a
    # concurrent request that reaches SciPy never sees a half-built torch.
    ensure_torch()
    if _load_model() is not None:
        return {"ready": True, **_load_meta()}
    stats = train_model(force=force)
    # _load_model is memoised, and the call above memoised a None from before
    # training. Without this the freshly trained weights stay invisible for the
    # lifetime of the process.
    _load_model.cache_clear()
    return {"ready": is_model_ready(), **stats}


def _label_masks(inventory: dict) -> dict:
    """word -> global indices of its own senses, used to mask predictions."""
    masks: dict[str, set[int]] = {}
    labels = inventory["labels"]
    for (word, _name), index in labels.items():
        masks.setdefault(word, set()).add(index)
    return masks


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
def _target_word(word: str) -> str | None:
    """Map an inflected token back to a word in :data:`TARGET_VOCAB`.

    Comments inflect: "I dropped my phone" has to reach the "drop" inventory,
    and "a bug in the script" the "bug" entry.
    """
    if word in TARGET_VOCAB_SET:
        return word
    lemma = _lemma(word)
    if lemma in TARGET_VOCAB_SET:
        return lemma
    for candidate in (word[:-2] if word.endswith("es") else word[:-1],):
        if candidate in TARGET_VOCAB_SET:
            return candidate
    return None


@lru_cache(maxsize=1)
def _lemma(word: str) -> str:
    ensure_nltk()
    from nltk.stem import WordNetLemmatizer

    for pos in ("n", "v", "a"):
        candidate = WordNetLemmatizer().lemmatize(word, pos)
        if candidate != word and "_" not in candidate:
            return candidate
    return word


def disambiguate(text: str) -> list[dict]:
    """Resolve every ambiguous word in ``text`` to a WordNet sense."""
    if not text or not text.strip():
        return []
    ensure_nltk()

    inventory = _sense_inventory()["by_word"]
    loaded = _load_model()
    sentences = [s for s in text.split(SENTENCE_BOUNDARY) if s.strip()] or [text]

    results: list[dict] = []

    for sentence in sentences:
        words = _WORD_RE.findall(sentence.lower())
        seen: set[str] = set()
        for position, raw in enumerate(words):
            word = _target_word(raw)
            if word is None or word in seen:
                continue
            seen.add(word)

            window = words[max(0, position - 6) : position] + words[position + 1 : position + 7]
            distribution = _predict(word, window, loaded)
            if not distribution:
                name, gloss, confidence = _gloss_overlap(word, set(window))
                if not name:
                    continue
                results.append(
                    {
                        "word": raw,
                        "sense": name,
                        "gloss": _short_gloss(gloss),
                        "confidence": confidence,
                        "margin": 0.0,
                        "source": "gloss-overlap",
                        "alternatives": [],
                    }
                )
                continue

            best_index = max(distribution, key=distribution.get)
            spread = sorted(distribution.values(), reverse=True)
            confidence = spread[0]
            margin = spread[0] - (spread[1] if len(spread) > 1 else 0.0)
            name, gloss = _sense_labels()[(word, best_index)]

            results.append(
                {
                    "word": raw,
                    "sense": name,
                    "gloss": _short_gloss(gloss),
                    "confidence": round(confidence, 3),
                    "margin": round(margin, 3),
                    "source": "gru",
                    "alternatives": [
                        {
                            "sense": _sense_labels()[(word, index)][0],
                            "gloss": _short_gloss(_sense_labels()[(word, index)][1]),
                            "score": round(score, 3),
                        }
                        for index, score in sorted(
                            distribution.items(), key=lambda kv: -kv[1]
                        )[:3]
                    ],
                }
            )
    return results



@lru_cache(maxsize=1)
def _sense_labels() -> dict:
    """(word, global label) -> (synset name, full gloss)."""
    inventory = _sense_inventory()
    labels = {}
    for (word, name), index in inventory["labels"].items():
        for sense in inventory["by_word"][word]:
            if sense["name"] == name:
                labels[(word, index)] = (name, sense["gloss"])
                break
    return labels


def _predict(word: str, window: list[str], loaded) -> dict:
    """Sense distribution for one occurrence, masked to the word's own senses."""
    if loaded is None:
        return {}
    model, vocabulary, masks, _meta = loaded
    allowed = masks.get(word)
    if not allowed:
        return {}

    import torch

    tokens = [vocabulary.get(f"<{word}>", 0)] + [vocabulary.get(t, 0) for t in window]
    tokens = tokens[:MAX_CONTEXT]
    tensor = torch.tensor([tokens], dtype=torch.long)
    with torch.no_grad():
        logits = model(tensor, torch.tensor([len(tokens)]))[0]
    scores = torch.softmax(logits, dim=0)

    # Combine the network's belief with how much of the comment each sense's
    # own WordNet vocabulary actually explains. Without this the GRU has only a
    # few hundred training contexts to learn from, and a single matching word
    # such as "river" is too diluted by the padding to move it.
    signals = _sense_vocabulary()
    labels = _sense_labels()
    context_words = set(window)
    denominator = len(context_words) or 1

    # Additive in log space, so each term is a readable, tunable weight:
    #   log P(sense)  +  LEXICAL_PRIOR * share of comment explained  +  cue hits
    combined: dict[int, float] = {}
    for index in allowed:
        name = labels[(word, index)][0]
        sense_words = signals.get((word, index), set())
        # Share of the *comment* this sense explains, not share of the sense's
        # own vocabulary it explains. The latter rewards senses with a short
        # definition and makes a rare sense an attractor.
        overlap = len(context_words & sense_words) / denominator
        cues = SENSE_CUES.get(word, {}).get(name)
        hits = len(context_words & cues) if cues else 0
        score = (
            math.log(float(scores[index]) + 1e-6)
            + LEXICAL_PRIOR * overlap
            + CUE_WEIGHT * min(hits, 3)
        )
        combined[index] = score

    best = max(combined.values())
    weights = {index: math.exp(score - best) for index, score in combined.items()}
    total = sum(weights.values()) or 1.0
    return {index: weight / total for index, weight in weights.items()}


def wsd_report(texts: list[str], top_k: int = 8) -> dict:
    """Aggregate sense choices across comments."""
    rows: list[dict] = []
    counts: Counter = Counter()
    by_word: dict[str, Counter] = {}
    for text in texts:
        for item in disambiguate(text):
            rows.append(item)
            counts[item["word"]] += 1
            by_word.setdefault(item["word"], Counter())[item["sense"]] += 1

    meta = _load_meta()
    model_info = {
        "trained": bool(CACHE_PATH.exists()),
        "cached": bool(CACHE_PATH.exists()),
        "words": meta.get("words", 0),
        "examples": meta.get("examples", 0),
        "parameters": meta.get("parameters", 0),
        "train_accuracy": meta.get("train_accuracy"),
        "final_loss": meta.get("final_loss"),
        "engine": "GRU" if _load_model() is not None else "gloss-overlap fallback",
    }

    inventory_index = _sense_inventory()["by_word"]
    ambiguous = []
    for word, total in counts.most_common(top_k):
        senses = by_word[word]
        top_sense, top_count = senses.most_common(1)[0]
        gloss = ""
        # The inventory is keyed by lemma, so "windows" has to be folded back
        # to "window" before the gloss can be looked up.
        inventory = inventory_index.get(word) or inventory_index.get(_target_word(word)) or []
        for sense in inventory:
            if sense["name"] == top_sense:
                gloss = _short_gloss(sense["gloss"])
                break
        ambiguous.append(
            {
                "word": word,
                "occurrences": total,
                "senses_used": len(senses),
                "top_sense": top_sense,
                "top_gloss": gloss,
                "agreement": round(top_count / total, 3),
                "breakdown": [
                    {"sense": name, "count": count, "share": round(count / total, 3)}
                    for name, count in senses.most_common(4)
                ],
            }
        )

    return {
        "disambiguated": len(rows),
        "words": len(counts),
        "ambiguous_words": [row["word"] for row in ambiguous],
        "rows": ambiguous,
        "examples": rows[:12],
        "model": model_info,
    }


def is_model_ready() -> bool:
    return CACHE_PATH.exists() and bool(_load_meta())


__all__ = [
    "CACHE_PATH",
    "TARGET_VOCAB",
    "disambiguate",
    "ensure_torch",
    "is_model_ready",
    "train_model",
    "warm",
    "wsd_report",
]
