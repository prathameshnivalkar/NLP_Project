from .chunking import chunk_report, chunk_sentence, chunk_text, feature_selection_study
from .morphology import analyze_text, analyze_word, generate_word_forms, morphology_report
from .ner import entity_report, extract_entities
from .ngram import generate_ngrams, ngram_report, phrase_trends
from .pipeline import analyze_video
from .postagger import compare_tagger, pos_report, tag_text
from .similarity import duplicate_rate, group_similar, similarity_report
from .wsd import disambiguate, wsd_report

__all__ = [
    "analyze_text",
    "analyze_video",
    "analyze_word",
    "chunk_report",
    "chunk_sentence",
    "chunk_text",
    "compare_tagger",
    "disambiguate",
    "duplicate_rate",
    "entity_report",
    "extract_entities",
    "feature_selection_study",
    "generate_ngrams",
    "generate_word_forms",
    "group_similar",
    "morphology_report",
    "ngram_report",
    "phrase_trends",
    "pos_report",
    "similarity_report",
    "tag_text",
    "wsd_report",
]
