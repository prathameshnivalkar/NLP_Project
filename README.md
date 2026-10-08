# Signal Lab

Signal Lab is an interactive Flask dashboard for analyzing YouTube comment sentiment and audience trends. The interface uses semantic HTML, CSS variables, and vanilla JavaScript, with an animated dark/light theme inspired by the visual language of Watermelon UI.

## Run locally

```bash
cd /home/ubuntu/youtube-comment-analyzer
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env      # add your YouTube Data API key
python app.py
```

Open `http://127.0.0.1:5000` in a browser.

Startup takes roughly 15 seconds on Windows once the Experiment 10 model is cached.
Importing PyTorch and reading the WordNet index are both one-off costs, and they are
paid before the server starts accepting requests. On a machine with no `models/`
cache the first start also trains the Experiment 10 GRU, which adds about a minute;
the result is written to `models/` and later starts skip it. On Linux and macOS
expect a few seconds instead.

Run `python app.py` in debug mode without the auto-reloader. The reloader
re-executes the module, which would repeat the warm-up and the one-off training.
Restart manually after editing.

## How it works

`POST /api/analyze` takes a YouTube URL, validates it, and hands the video id to
`nlp/pipeline.py`. The pipeline fetches comments through the YouTube Data API and
falls back to deterministic seeded demo data when no key is configured or the
API is unreachable, so the dashboard is fully usable without credentials.

The expensive deep analysis runs on a sentiment-stratified sample of at most 250
comments (`LAB_SAMPLE` in `nlp/pipeline.py`); trend, topic and keyword figures
use the full comment set. The first analysis after startup takes about 12 seconds,
mostly waiting on the YouTube API; later ones take roughly one second.

All startup work happens on one thread on purpose. NLTK's WordNet reader is not
thread safe, so training the Experiment 10 model in the background made any
concurrent WordNet read fail with `assert self.fp is None` and surfaced as a 500
from `/api/morphology/forms`.

### Endpoints

| Route | Purpose |
| --- | --- |
| `GET /` | Dashboard |
| `POST /api/analyze` | Analyse a YouTube URL, returns the full payload |
| `GET /api/health` | Liveness, returns `{"service", "status"}` |
| `GET /api/morphology/forms` | Word-form generator used by the Method panel |

## Analysis modules

| Module | Covers | What it does |
| --- | --- | --- |
| `nlp/sentiment.py` | Exp 01 | Rule-based positive/neutral/negative scoring |
| `nlp/topics.py` | Exp 02 | LDA topic detection |
| `nlp/keywords.py` | Exp 03 | Weighted keyword extraction |
| `nlp/morphology.py` | Exp 04 | Word-form generation and root analysis |
| `nlp/ngram.py` | Exp 05 | Bigram/trigram extraction and phrase trends |
| `nlp/postagger.py` | Exp 06 | Three POS taggers, compared for agreement |
| `nlp/chunking.py` | Exp 07 | NP/VP/ADJP grammar and a feature-selection study |
| `nlp/ner.py` | Exp 08 | Domain gazetteer NER with context rules |
| `nlp/similarity.py` | Exp 09 | TF-IDF topic grouping and redundancy stats |
| `nlp/wsd.py` | Exp 10 | GRU word-sense disambiguation over 48 ambiguous words |

### Experiment 08 notes

NLTK 3.10 has no usable statistical NER model here: `NEChunkParser`,
`LookaheadTagger`, and the `maxent_ne_chunker_eng` download are all
unavailable, so `nlp/ner.py` is a domain gazetteer with phrase, honorific and
context rules instead. Single-word entities must be capitalised by the commenter,
which is what keeps the company `Apple` apart from `apple pie`. Capitalised
words that turn out to have ordinary English meanings are rejected rather than
guessed at, so sentence case never becomes a bogus entity.

### Experiment 10 notes

`nlp/wsd.py` trains a small GRU with masked mean pooling and a shared sense
inventory, supervised by WordNet definitions, examples, lemmas and hypernyms.
The trained weights are cached in `models/` (git-ignored), and the first run after
a config change trains them at startup, which takes about a minute. If training
fails for any reason the module answers with a gloss-overlap baseline and the
dashboard labels the sense panel as degraded, so nothing blocks. PyTorch is
imported eagerly at startup because SciPy reads `torch.Tensor` off the module
object without taking the import lock, which otherwise crashes the topic model
with a partially initialised module.

## Verifying

```bash
python check_experiments.py     # experiments 06-10, pipeline contract, performance
node _check_ui.js               # dashboard renderers against a stubbed DOM
```

`check_experiments.py` asserts the documented examples rather than current
behaviour: the Experiment 09 comments must group under `Audio Quality`, and
`bank` must read as a financial institution in one sentence and a river bank in
another. It exits non-zero on failure, so it works as a pre-commit gate without
needing a test framework installed.
