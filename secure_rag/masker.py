import re #regular expression module for pattern matching

def mask_text(text: str) -> str:#input == text 
    text = re.sub(r'\S+@\S+', '[EMAIL_MASKED]', text)#regex for email masking 
    text = re.sub(r'\b\d{10}\b', '[PHONE_MASKED]', text)#find & replace 10 digits pattern matched digit 
    return text#output == masked verison 