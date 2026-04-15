import pytest
from secure_rag.masker import mask_text


class TestRegexMasking:
    def test_email_masking(self):
        result = mask_text("contact dr. at doctor@hospital.com")
        assert "[EMAIL_MASKED]" in result
        assert "doctor@hospital.com" not in result

    def test_multiple_emails(self):
        result = mask_text("email a@b.com and c@d.com")
        assert result.count("[EMAIL_MASKED]") == 2
        assert "a@b.com" not in result

    def test_phone_10_digit(self):
        result = mask_text("call 9876543210")
        assert "[PHONE_MASKED]" in result

    def test_phone_multiple(self):
        result = mask_text("home: 9876543210 mobile: 9123456789")
        assert result.count("[PHONE_MASKED]") == 2

    def test_aadhaar_unspaced(self):
        result = mask_text("Aadhaar: 123456789012")
        assert "[AADHAAR_MASKED]" in result
        assert "123456789012" not in result

    def test_aadhaar_spaced(self):
        result = mask_text("Aadhaar: 1234 5678 9012")
        assert "[AADHAAR_MASKED]" in result

    def test_aadhaar_partial_false_positive(self):
        result = mask_text("case number 12345678901 and ref 123456789012")
        assert result.count("[AADHAAR_MASKED]") == 1
        assert "12345678901" not in result

    def test_pan_format(self):
        result = mask_text("PAN: ABCDE1234F")
        assert "[PAN_MASKED]" in result
        assert "ABCDE1234F" not in result

    def test_pan_multiple(self):
        result = mask_text("doctor PAN ABCDE1234F, patient PAN FGHIJ5678K")
        assert "ABCDE1234F" not in result
        assert "FGHIJ5678K" not in result

    def test_mrn_format(self):
        result = mask_text("MRN: MRN123456")
        assert "[PATIENT_ID_MASKED]" in result
        assert "MRN123456" not in result

    def test_uhid_format(self):
        result = mask_text("UHID12345")
        assert "[PATIENT_ID_MASKED]" in result

    def test_pid_format(self):
        result = mask_text("PID87654321")
        assert "[PATIENT_ID_MASKED]" in result

    def test_health_id_14_digit(self):
        result = mask_text("ABDM Health ID: 12345678901234")
        assert "[HEALTH_ID_MASKED]" in result

    def test_health_id_short(self):
        result = mask_text("Hospital ID: 12345678")
        assert "[HEALTH_ID_MASKED]" in result

    def test_health_id_false_positive_small(self):
        result = mask_text("room 302 and ward 1234567")
        assert "[HEALTH_ID_MASKED]" not in result

    def test_dob_ddmmyyyy(self):
        result = mask_text("DOB: 15/08/1990")
        assert "[DOB_MASKED]" in result

    def test_dob_ddmmyyyy_dashes(self):
        result = mask_text("DOB 25-12-1985")
        assert "[DOB_MASKED]" in result

    def test_dob_iso(self):
        result = mask_text("Date: 1990-08-15")
        assert "[DOB_MASKED]" in result

    def test_empty_string(self):
        result = mask_text("")
        assert result == ""

    def test_no_pii(self):
        result = mask_text("Patient presents with fever and cough.")
        assert result == "Patient presents with fever and cough."

    def test_mixed_pii(self):
        result = mask_text(
            "Patient John Doe, MRN123456, phone 9876543210, "
            "email patient@email.com, Aadhaar 1234 5678 9012, "
            "DOB 01/01/1990."
        )
        assert "MRN123456" not in result
        assert "9876543210" not in result
        assert "patient@email.com" not in result
        assert "1234 5678 9012" not in result
        expected_tokens = [
            "[EMAIL_MASKED]",
            "[PHONE_MASKED]",
            "[AADHAAR_MASKED]",
            "[PATIENT_ID_MASKED]",
            "[DOB_MASKED]",
        ]
        masked_count = sum(1 for t in expected_tokens if t in result)
        assert masked_count >= 5, f"Expected >= 5 mask tokens, got {masked_count}: {result}"


@pytest.mark.slow
class TestNERMasking:
    def test_person_name_masked(self):
        result = mask_text("Rajesh Kumar visited the clinic.")
        assert "[NAME_MASKED]" in result
        assert "Rajesh Kumar" not in result

    def test_gpe_city_masked(self):
        result = mask_text("Patient is from Mumbai.")
        assert "[ADDRESS_MASKED]" in result
        assert "Mumbai" not in result

    def test_org_hospital_masked(self):
        result = mask_text("Admitted to Apollo Hospital.")
        assert "[ORG_MASKED]" in result
        assert "Apollo" not in result
