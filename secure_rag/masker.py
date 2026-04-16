import re
import os

_nlp = None


def _get_nlp():
    global _nlp
    if _nlp is None:
        import spacy
        model = os.getenv("SPACY_MODEL", "en_core_web_sm")
        _nlp = spacy.load(model)
    return _nlp


def mask_text(text: str) -> str:
    text = re.sub(r'\S+@\S+', '[EMAIL_MASKED]', text)
    text = re.sub(r'\b\d+\s+[A-Z][a-zA-Z]+(\s+[A-Z][a-zA-Z]+)*\s+(Road|Street|Avenue|Lane|Marg)\b', '[ADDRESS_MASKED]', text)
    text = re.sub(r'\b\d{10}\b', '[PHONE_MASKED]', text)
    text = re.sub(r'\b\d{4}\s?\d{4}\s?\d{4}\b', '[AADHAAR_MASKED]', text)
    text = re.sub(r'\b[A-Z]{5}[0-9]{4}[A-Z]\b', '[PAN_MASKED]', text)
    text = re.sub(r'\b\d{8,16}\b', '[HEALTH_ID_MASKED]', text)
    text = re.sub(r'\b(MRN|UHID|PID)[0-9]{5,14}\b', '[PATIENT_ID_MASKED]', text)
    text = re.sub(
        r'\b\d{2}[/-]\d{2}[/-]\d{4}\b|\b\d{4}[/-]\d{2}[/-]\d{2}\b',
        '[DOB_MASKED]',
        text,
    )

    try:
        nlp = _get_nlp()
        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                text = text.replace(ent.text, '[NAME_MASKED]', 1)
            elif ent.label_ in ("GPE", "LOC", "FAC"):
                text = text.replace(ent.text, '[ADDRESS_MASKED]', 1)
            elif ent.label_ == "ORG":
                text = text.replace(ent.text, '[ORG_MASKED]', 1)
    except Exception:
        pass

    return text
