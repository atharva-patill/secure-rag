from secure_rag.masker import mask_text


def test_email_masking():
    text = "contact me at test@gmail.com"
    result = mask_text(text)
    assert "[EMAIL_MASKED]" in result


def test_phone_masking():
    text = "call me at 9876543210"
    result = mask_text(text)
    assert "[PHONE_MASKED]" in result