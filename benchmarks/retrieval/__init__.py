"""
Retrieval evaluation framework for Secure RAG.

Measures retrieval quality independently from LLM generation.
Consumes ground truth from ground_truth.py and shared utilities from _common.py.
"""

from . import ground_truth
