import os
from pathlib import Path

import pytest

from secure_rag import build_rag


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "pii_medical.txt"

PII_PATTERNS = [
    "1234 5678 9012",
    "9876543210",
    "rajesh.kumar@email.com",
    "ABCDE1234F",
    "UHID123456789",
    "MRN2024001",
    "12345678901234",
    "15/08/1990",
    "42 MG Road",
    "Rajesh Kumar",
    "Mumbai",
    "Apollo",
]


class TestPreEmbeddingMasking:
    def test_no_raw_pii_in_chunks(self):
        if not FIXTURE_PATH.exists():
            pytest.skip("Fixture file not found")

        os.environ.setdefault("HF_TOKEN", "dummy-for-test")
        os.environ.setdefault("SPACY_MODEL", "en_core_web_sm")

        vector_store, chunks = build_rag(str(FIXTURE_PATH))

        all_text = " ".join(chunks)
        for pattern in PII_PATTERNS:
            assert pattern not in all_text, f"Raw PII leaked: '{pattern}'"

    def test_masked_tokens_present(self):
        if not FIXTURE_PATH.exists():
            pytest.skip("Fixture file not found")

        os.environ.setdefault("HF_TOKEN", "dummy-for-test")
        os.environ.setdefault("SPACY_MODEL", "en_core_web_sm")

        vector_store, chunks = build_rag(str(FIXTURE_PATH))

        all_text = " ".join(chunks)
        expected_tokens = [
            "[AADHAAR_MASKED]",
            "[PHONE_MASKED]",
            "[EMAIL_MASKED]",
            "[PAN_MASKED]",
            "[PATIENT_ID_MASKED]",
            "[HEALTH_ID_MASKED]",
            "[DOB_MASKED]",
        ]
        for token in expected_tokens:
            assert token in all_text, f"Missing masked token: '{token}'"

    def test_empty_file_raises(self):
        tmp = Path("/tmp/empty_test_rag.txt")
        tmp.write_text("")
        try:
            with pytest.raises(ValueError):
                build_rag(str(tmp))
        finally:
            tmp.unlink(missing_ok=True)
