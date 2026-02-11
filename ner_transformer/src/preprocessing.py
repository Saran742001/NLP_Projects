import re

def clean_text(text: str) -> str:
    """
    Light text cleaning for NER
    """
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text
