"""Verification for Experiments 06-10 and their dashboard wiring.

Run with ``python check_experiments.py``. Exits non-zero on the first failure, so
it works as a pre-commit gate without needing a test framework installed.

The expectations here are the ones the documentation in exp6-10.txt states, not
whatever the code happens to produce: the Exp09 example must group three
differently worded comments under "Audio Quality", and the Exp10 example must
read "bank" as a financial institution in one sentence and a river bank in
another.
"""
from __future__ import annotations

import json
import time

from nlp.chunking import chunk_sentence, feature_selection_study
from nlp.ner import extract_entities
from nlp.pipeline import analyze_comments
from nlp.postagger import TAGGERS, pos_report, tag_text, tag_with
from nlp.preprocessor import tokenize_full_batch
from nlp.similarity import group_similar, similarity_report
from nlp.wsd import wsd_report

failures: list[str] = []


def check(label: str, condition: bool, detail: object = "") -> None:
    if condition:
        print(f"  ok   {label}")
    else:
        print(f"  FAIL {label} {detail!r}")
        failures.append(label)


# --------------------------------------------------------------------------
# Experiment 06 - POS tagging
# --------------------------------------------------------------------------
print("== Experiment 06: POS tagging")
sentence = "The quick brown foxes jumped over the lazy dog"
tags = dict(tag_text(sentence))
check("every token receives a tag", all(tag for _, tag in tag_text(sentence)))
check("determiners are DT", tags.get("the") == "DT", tags)
check("adjectives are JJ", tags.get("quick") == "JJ", tags)
check("nouns are tagged NNS when plural", tags.get("foxes") == "NNS", tags)
check("verbs are VBD when past", tags.get("jumped") == "VBD", tags)

check("three taggers are offered", set(TAGGERS) == {"perceptron", "rules", "hybrid"}, TAGGERS)
batched = tokenize_full_batch([sentence, sentence])
check("tagging is per sentence", len(tag_with(batched, "perceptron")) == 2)
check(
    "each tagger gives its own output",
    tag_with(batched, "rules") != tag_with(batched, "perceptron"),
)

report = pos_report(batched)
check("report counts tokens", report["tokens"] > 0)
check("report groups tags into plain English", bool(report["distribution"]))
check(
    "report compares the taggers",
    len(report["tagger_comparison"]) == 3,
    report["tagger_comparison"],
)
check(
    "baseline agrees with itself at 100%",
    any(
        row["tagger"] == report["baseline"] and row["agreement"] == 100.0
        for row in report["tagger_comparison"]
    ),
    report["tagger_comparison"],
)
check("an empty input yields an empty report", pos_report([[]])["distribution"] == [])

# --------------------------------------------------------------------------
# Experiment 07 - chunking and the feature-selection study
# --------------------------------------------------------------------------
print("\n== Experiment 07: chunking and feature selection")
chunks = chunk_sentence(tokenize_full_batch([sentence])[0])
check("phrases are found", bool(chunks), chunks)
check("phrase text is preserved verbatim", all(chunk["text"] for chunk in chunks))
check(
    "phrases carry a readable label",
    all(chunk["label"] in {"Noun Phrase", "Verb Phrase", "Adjective Phrase"} for chunk in chunks),
    chunks,
)

CORPUS = [
    "The audio quality is very poor and the sound is terrible",
    "great tutorial very helpful thanks",
    "this tutorial is really helpful thank you",
    "audio is really bad in this video",
    "i cannot hear the audio properly at all",
    "the subtitle timing is completely wrong here",
    "thanks for this great tutorial honestly",
    "excellent walkthrough really useful for beginners",
    "the script does not run on my machine",
    "very helpful and clear explanation of python",
    "that intro music is really annoying",
    "please show the keyboard shortcuts for windows",
    "the microphone sounds great in this episode",
    "what a fantastic and useful video tutorial",
    "the export settings are broken in this build",
    "i hate how quiet the recording is here",
    "clear explanation and great pacing throughout",
    "please add more examples to each section",
    "the audio quality is really poor overall",
    "this tutorial is extremely useful and great",
]
LABELS = [
    "negative", "positive", "positive", "negative", "negative", "negative", "positive",
    "positive", "negative", "positive", "negative", "neutral", "positive", "positive",
    "negative", "negative", "positive", "neutral", "negative", "positive",
]

study = feature_selection_study(tokenize_full_batch(CORPUS), LABELS)
check("the study compares four feature sets", len(study["feature_sets"]) == 4)
check("the study produces rows", bool(study["rows"]), study.get("reason"))
check(
    "every row reports accuracy, F1 and vocabulary",
    all(
        {"feature_set", "fraction", "train_size", "accuracy", "f1", "vocabulary"} <= set(row)
        for row in study["rows"]
    ),
)
check("the study names a best combination", isinstance(study["best"], dict) and "f1" in study["best"])
check(
    "a corpus that is too small says why",
    feature_selection_study(tokenize_full_batch(["ok"]), ["positive"])["rows"] == [],
)

# --------------------------------------------------------------------------
# Experiment 08 - named entity recognition
# --------------------------------------------------------------------------
print("\n== Experiment 08: named entity recognition")
ENTITIES_THAT_MUST_APPEAR = [
    ("I learned Python from this tutorial", "Python", "Technology"),
    ("Microsoft released a new update", "Microsoft", "Organization"),
    ("Dr Sarah Chen explained the model", "Dr Sarah Chen", "Person"),
    ("Sarah Chen uploaded this", "Sarah Chen", "Person"),
    ("We visited London and Paris last summer", "London", "Location"),
    ("We visited London and Paris last summer", "Paris", "Location"),
    ("This DaVinci Resolve tutorial is great", "DaVinci Resolve", "Product"),
    ("Nvidia released faster GPUs for deep learning", "GPUs", "Technology"),
    ("Acme Inc partnered with Stanford University", "Acme Inc", "Organization"),
    ("Acme Inc partnered with Stanford University", "Stanford University", "Organization"),
    ("I use Visual Studio Code", "Visual Studio Code", "Product"),
    ("Maria Gonzalez made this", "Maria Gonzalez", "Person"),
]
for text, expected, entity_type in ENTITIES_THAT_MUST_APPEAR:
    found = [(e["text"], e["type"]) for e in extract_entities(text)]
    check(
        f"{expected!r} recognised as {entity_type}",
        (expected, entity_type) in found,
        found,
    )

# A capitalised word is usually just sentence case, not a name. These were all
# inventing "Organization" entities before the name-pattern rule was tightened.
SENTENCES_THAT_MUST_STAY_CLEAN = [
    "Monday I watched this",
    "The Tutorial Is Great",
    "I Bought This Yesterday",
    "Best Tutorial Ever",
    "Please Fix The Export",
    "This Is A Test Comment",
    "Thanks For The Video",
    "Really Good Explanations",
    "Fresh From The Room",
    "Please Subscribe To The Channel",
    "First Video Here",
    "To Be Honest",
    "In My Opinion",
    # Lowercase brand names stay ordinary words, and the fruit is not the company.
    "my microphone sounds great",
    "i had apple pie for dessert",
    "the bank should improve its mobile app",
]
for text in SENTENCES_THAT_MUST_STAY_CLEAN:
    found = extract_entities(text)
    check(f"no entity invented in {text!r}", not found, found)

check(
    "a surname does not break the given-name rule",
    any(
        e["text"] == "Sarah Chen" and e["source"] == "name-pattern"
        for e in extract_entities("Sarah Chen uploaded this")
    ),
)
check(
    "two given names skip the given-name rule",
    not any(
        e["source"] == "name-pattern" for e in extract_entities("Mary Anne made this")
    ),
    extract_entities("Mary Anne made this"),
)

# --------------------------------------------------------------------------
# Experiment 09 - real-time text similarity
# --------------------------------------------------------------------------
print("\n== Experiment 09: text similarity")
DOC_EXAMPLE = [
    "The audio quality is very poor.",
    "The sound quality is bad.",
    "I cannot hear the audio properly.",
]
grouped = group_similar(DOC_EXAMPLE, ["negative"] * 3)
check("the documented three comments form one group", len(grouped) == 1, grouped)
check(
    "the documented group is called Audio Quality",
    grouped and grouped[0]["label"] == "Audio Quality",
    grouped and grouped[0]["label"],
)
check(
    "all three documented comments are kept",
    grouped and grouped[0]["size"] == 3,
    grouped and grouped[0]["size"],
)
check(
    "the group's sentiment is negative",
    grouped and grouped[0]["sentiment"] == "negative",
    grouped and grouped[0]["sentiment"],
)

# Sharing a mood is not the same as sharing a topic.
off_topic = group_similar(
    [
        "The audio quality is very poor and the sound is terrible",
        "The pricing page is confusing and unclear",
        "I cannot hear the audio at all",
    ],
    ["negative", "negative", "negative"],
)
absorbed = [member["text"] for row in off_topic for member in row["members"]]
check("an off-topic complaint is not absorbed", not any("pricing" in t for t in absorbed), absorbed)
check("only the two on-topic comments group", off_topic and off_topic[0]["size"] == 2, off_topic)

check(
    "subjects tied on frequency keep document order",
    group_similar(
        ["The audio quality is poor", "The quality of this audio is low"], ["negative"] * 2
    )[0]["label"] == "Audio Quality",
)
check(
    "blank comments do not shift the cohesion maths",
    (
        lambda rows: rows and rows[0]["size"] == 3 and rows[0]["cohesion"] > 0
    )(
        group_similar(
            [
                "",
                "The audio quality is very poor",
                "   ",
                "The sound quality is bad",
                "I cannot hear the audio",
            ],
            ["negative"] * 5,
        )
    ),
)
check("no comments is not an error", group_similar([], []) == [])
check("a report over the corpus works", similarity_report(CORPUS, LABELS)["groups"] >= 1)

# --------------------------------------------------------------------------
# Experiment 10 - word sense disambiguation
# --------------------------------------------------------------------------
print("\n== Experiment 10: word sense disambiguation")
wsd = wsd_report([
    "The bank should improve its mobile app",
    "The video shows people sitting near the river bank",
])
bank = next((row for row in wsd["rows"] if row["word"] == "bank"), None)
check("bank is treated as ambiguous", bank is not None, wsd["rows"])
check("both readings of bank appear", bank and bank["senses_used"] == 2, bank and bank["senses_used"])
check(
    "the app is read as a financial institution",
    bank and any("depository" in sense["sense"] for sense in bank["breakdown"]),
    bank and bank["breakdown"],
)
check(
    "the river bank is read as a river bank",
    bank and any(sense["sense"].startswith("bank.") for sense in bank["breakdown"]),
    bank and bank["breakdown"],
)
check("the engine in use is reported", wsd["model"]["engine"], wsd["model"])
check("no comments is not an error", wsd_report([])["rows"] == [])

# Inflected forms have to fold back to the lemma before the sense is looked up;
# "windows" is not a key in the inventory, "window" is.
inflected = wsd_report(["please show the keyboard shortcuts for windows"])
check(
    "an inflected target word does not crash the report",
    bool(inflected["rows"]) and all(row["top_sense"] and row["top_gloss"] for row in inflected["rows"]),
    inflected["rows"],
)

# --------------------------------------------------------------------------
# Pipeline and payload contract
# --------------------------------------------------------------------------
print("\n== Pipeline payload")
COMMENTS = [
    {"text": text, "published_at": f"2026-09-{(index % 28) + 1:02d}T10:00:00Z"}
    for index, text in enumerate(CORPUS)
]
analysis = analyze_comments(COMMENTS)
for key in ("pos", "chunks", "feature_study", "entities", "similarity", "wsd", "metrics", "summary"):
    check(f"payload carries {key}", key in analysis)
check("the payload is JSON serialisable", bool(json.dumps(analysis)))

# --------------------------------------------------------------------------
# Performance budget
# --------------------------------------------------------------------------
print("\n== Performance")
start = time.perf_counter()
group_similar(CORPUS * 250, (LABELS * 250)[: len(CORPUS) * 250])
elapsed = time.perf_counter() - start
check(f"5000 comments grouped in {elapsed:.2f}s (budget 6s)", elapsed < 6.0)

start = time.perf_counter()
analyze_comments(COMMENTS * 10)
elapsed = time.perf_counter() - start
check(f"200 comments analysed in {elapsed:.2f}s (budget 8s)", elapsed < 8.0)

print()
if failures:
    print(f"{len(failures)} CHECK(S) FAILED: {', '.join(failures)}")
    raise SystemExit(1)
print("ALL CHECKS PASSED")
