import pandas as pd
import re
from indic_transliteration import sanscript

def detect_script(text: str) -> str:
    if not isinstance(text, str):
        return None
        
    if re.search(r'[\u0900-\u097F]', text):
        return sanscript.DEVANAGARI
    elif re.search(r'[\u0A80-\u0AFF]', text):
        return sanscript.GUJARATI
    elif re.search(r'[\u0980-\u09FF]', text):
        return sanscript.BENGALI
    elif re.search(r'[\u0B80-\u0BFF]', text):
        return sanscript.TAMIL
    return None

def normalize_script(text: str) -> str:
    if pd.isna(text) or not isinstance(text, str):
        return ""
        
    script = detect_script(text)
    if script:
        # Transliterate to HK (Harvard-Kyoto) or ITRANS
        text = sanscript.transliterate(text, script, sanscript.ITRANS)
    
    return text
