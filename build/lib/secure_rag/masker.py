import re

def mask_text(text: str) -> str:
    text = re.sub(r'\S+@\S+', '[EMAIL_MASKED]', text)
    text = re.sub(r'\b\d{10}\b', '[PHONE_MASKED]', text)
    return text