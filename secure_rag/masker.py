from pyexpat import model
import re #regular expression module for pattern matching
import os
_nlp = None

def _get_nlp():
    global _nlp
    if _nlp is None:
        import spacy
        model = os.getenv("SPACY_MODEL","en_core_web_sm")
        _nlp=spacy.load(model)
        return _nlp
        
def mask_text(text: str) -> str:#input == text 
    text = re.sub(r'\S+@\S+', '[EMAIL_MASKED]', text)#regex for email masking 
    text = re.sub(r'\b\d{10}\b', '[PHONE_MASKED]', text)#find & replace 10 digits pattern matched digit 
    return text#output == masked verison 