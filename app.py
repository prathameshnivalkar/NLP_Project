from __future__ import annotations

import re
from dotenv import load_dotenv
from urllib.parse import parse_qs, urlparse

load_dotenv()

from flask import Flask, jsonify, render_template, request
from nlp.morphology import analyze_word, generate_word_forms
from nlp.pipeline import analyze_video

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False

YOUTUBE_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be"}

MAX_WORD_LENGTH = 40


def extract_video_id(url: str) -> str | None:
    try:
        parsed = urlparse(url.strip())
    except ValueError:
        return None

    if parsed.netloc.lower() not in YOUTUBE_HOSTS:
        return None

    if parsed.netloc.lower() == "youtu.be":
        video_id = parsed.path.strip("/").split("/")[0]
    else:
        video_id = parse_qs(parsed.query).get("v", [None])[0]

    if not video_id or not re.fullmatch(r"[A-Za-z0-9_-]{6,20}", video_id):
        return None
    return video_id


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/health")
def health():
    return jsonify({"status": "ok", "service": "signal-lab"})


@app.get("/api/morphology/forms")
def morphology_forms():
    """Experiment 04 playground: decompose a word and generate its word forms."""
    word = str(request.args.get("word", "")).strip().lower()
    if not word or len(word) > MAX_WORD_LENGTH or not word.isalpha():
        return jsonify({"error": "Provide a single alphabetic word to analyze."}), 400

    analysis = analyze_word(word)
    if not analysis.get("root"):
        return jsonify({"error": f"Could not analyze '{word}'. Try a common English word."}), 404

    base = analysis.get("root") or analysis.get("lemma") or word
    return jsonify({"word": word, "analysis": analysis, "forms": generate_word_forms(base)})


@app.post("/api/analyze")
def analyze():
    payload = request.get_json(silent=True) or {}
    url = str(payload.get("url", "")).strip()
    video_id = extract_video_id(url)
    if not video_id:
        return jsonify({"error": "Enter a valid public YouTube URL, such as https://www.youtube.com/watch?v=..."}), 400
    return jsonify(analyze_video(video_id, url))


def _load_expensive_corpora() -> None:
    """Pay the one-off corpus loads at startup, before any request arrives.

    Importing PyTorch and reading the WordNet index together cost about forty
    seconds on Windows. Doing that in a background thread did not help: the
    interpreter holds the GIL while loading either one, so the first request
    simply queued up behind the warm thread and then paid the same cost again.
    Warming here means startup is slow once and every request afterwards is fast.

    The Experiment 10 GRU used to train in a background thread too, and that was
    worse than slow. NLTK's WordNet reader is not thread safe - it caches open
    zip pointers on the instance - so while the trainer was reading WordNet,
    any request that also read it died with "AssertionError: assert self.fp is
    None" from inside nltk.data, which surfaced as a 500 from
    /api/morphology/forms. Nothing in this app needs a second thread, so the
    model is trained here instead. The weights are cached in models/, so this
    cost is paid once ever rather than on every start.

    If training fails for any reason, nlp.wsd keeps answering with its gloss
    baseline and the dashboard still works.
    """
    # Torch first, on the main thread. SciPy's array-API layer reads
    # torch.Tensor straight off the module object without taking the import
    # lock, so anything that reaches the LDA topic model while torch is still
    # importing fails with "partially initialized module 'torch' has no
    # attribute 'Tensor'".
    try:
        from nlp.wsd import ensure_torch

        ensure_torch()
    except Exception:
        pass

    try:
        from nlp.preprocessor import preprocess

        preprocess("warm up the morphological caches")
    except Exception:
        pass

    try:
        from nlp.wsd import is_model_ready, warm

        if not is_model_ready():
            print(
                "Training the Experiment 10 model. This happens once, takes about"
                " a minute, and is cached in models/ afterwards."
            )
        warm()
    except Exception as exc:  # noqa: BLE001
        print(f"Experiment 10 model unavailable, using the gloss baseline: {exc}")


_load_expensive_corpora()

if __name__ == "__main__":
    # No reloader: it re-executes this module, which would train the model twice
    # and pay the startup cost twice. Restart manually after editing.
    app.run(debug=True, use_reloader=False, host="127.0.0.1", port=5000)

