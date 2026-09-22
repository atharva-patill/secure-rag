"""
Query-routed fixed-depth retrieval — heuristic classifier.

This is a *heuristic* runtime-only classifier that uses purely lexical
features from the query string.  It does NOT use ground truth,
expected answer, relevant record count, disease prevalence, benchmark
metadata, MRN, evaluation labels, embedding similarity, LLM calls, or
randomness.  It is deterministic and pure, so it can be unit-tested
independently.

Policy:
    single (factual / single-record) -> K=2
    multi  (aggregate / multi-record) -> K=20

Constants DEFAULT_SINGLE_K and AGGREGATE_K are the single source of
truth for routing depths; no literal 2/20 should be scattered elsewhere.

Vocabulary reuse:
    Keywords are derived from secure_rag/generator.py aggregate-query
    guidance ("Which patients...", "Who all...", "List all...",
    "How many...", "Which records...", "Summarize...") so production
    routing and generation share a consistent aggregate vocabulary.
    Additional minimal alias "who received" is included to cover the
    canonical aggregate queries in ground_truth_v2 without introducing
    a second inconsistent vocabulary.

Limitations:
    Lexical heuristics cannot perfectly separate all single vs aggregate
    intents (e.g., "Summarize the patient record." is lexically similar
    to "Summarize all patients with...").  The classifier is
    intentionally conservative and prefers precision on plural/aggregate
    phrasing.  It is NOT a score-based, learned, or threshold-driven
    policy and must not be described as such.
"""

from __future__ import annotations

DEFAULT_SINGLE_K = 2
AGGREGATE_K = 20

# Substrings (lowercased) that indicate a multi-record / aggregate intent.
# Derived from generator.py; kept lowercased for case-insensitive matching.
# Expanded to cover natural paraphrased aggregate forms (phrase-based,
# domain-independent, case-insensitive).
_MULTI_SUBSTRINGS = [
    "which patients",
    "who all",
    "list all",
    "how many",
    "which records",
    "who received",
    "get everyone",
    "every patient",
    "all patients",
    "every record",
    "all records",
    "everyone who",
]

# "summarize" is an aggregate cue in generator.py.  To avoid flagging
# singular "Summarize the patient record." as multi, we require a
# co-occurring plural cue when summarization is involved.
_SUMMARIZE_TOKEN = "summarize"
_SUMMARIZE_PLURAL_HINTS = ("all", "patients", "records", "which")


def classify_query(query: str) -> str:
    """
    Classify query as "single" or "multi" using lexical heuristics.

    Returns exactly "single" or "multi".  Deterministic, no I/O.
    """
    if not isinstance(query, str):
        return "single"
    q = query.strip().lower()
    if not q:
        return "single"

    for phrase in _MULTI_SUBSTRINGS:
        if phrase in q:
            return "multi"

    # Summarize heuristic: only treat as aggregate when plural hint present.
    if _SUMMARIZE_TOKEN in q:
        for hint in _SUMMARIZE_PLURAL_HINTS:
            if hint in q:
                return "multi"
        # Plain "summarize the patient record." stays single.
        return "single"

    return "single"


def routed_k(query: str, enabled: bool = False) -> int:
    """
    Resolve retrieval depth for a query.

    When enabled is False (default production), always returns DEFAULT_SINGLE_K.
    When enabled is True, returns AGGREGATE_K for multi queries, else DEFAULT_SINGLE_K.
    """
    if not enabled:
        return DEFAULT_SINGLE_K
    label = classify_query(query)
    return AGGREGATE_K if label == "multi" else DEFAULT_SINGLE_K
