"""
Text and entity name normalization module.
"""
import re
import unicodedata
from typing import Dict, List, Set, Tuple
import jellyfish

# Generic stop words and legal suffix variants
LEGAL_SUFFIX_MAP = {
    r"\bpvt\.?\s*ltd\.?\b": "private limited",
    r"\bpvt\.?\b": "private",
    r"\bltd\.?\b": "limited",
    r"\bcorp\.?\b": "corporation",
    r"\binc\.?\b": "incorporated",
    r"\bllc\.?\b": "llc",
    r"\bl\.l\.c\.?\b": "llc",
    r"\bco\.?\b": "company",
    r"\bco\s+ltd\.?\b": "company limited",
}

LEGAL_SUFFIX_TOKENS = {
    "private", "limited", "pvt", "ltd", "corp", "corporation",
    "inc", "incorporated", "llc", "company", "co", "enterprises",
    "enterprise", "group", "holdings", "holding", "services",
    "solutions", "international", "global", "industries", "industry"
}

GENERIC_NAME_STOPWORDS = {
    "the", "and", "of", "in", "for", "at", "by", "a", "an", "&"
}


def unicode_clean(text: str) -> str:
    """Normalize unicode characters and strip accents."""
    if not text:
        return ""
    text = str(text)
    norm = unicodedata.normalize("NFKD", text)
    return "".join(c for c in norm if not unicodedata.combining(c))


def normalize_country(country: str) -> str:
    """
    Normalizes country string format (whitespace, casing) without hardcoding
    or limiting to any fixed vocabulary. Unseen countries like 'France' pass through.
    """
    if not country:
        return ""
    clean = unicode_clean(country).strip()
    return re.sub(r"\s+", " ", clean).upper()


def normalize_business_name(raw_name: str) -> Dict[str, any]:
    """
    Normalizes business name preserving raw text.
    Returns dictionary with normalized text, core tokens, legal suffixes,
    and phonetic encodings.
    """
    if not raw_name:
        return {
            "name_raw": "",
            "name_norm": "",
            "name_tokens": [],
            "core_tokens": [],
            "legal_suffix": "",
            "phonetic_key": "",
        }

    # 1. Unicode normalize & lowercase
    text = unicode_clean(raw_name).lower()

    # 2. Replace & with 'and'
    text = re.sub(r"&", " and ", text)

    # 3. Standardize legal suffixes
    legal_found = []
    for pattern, replacement in LEGAL_SUFFIX_MAP.items():
        if re.search(pattern, text):
            legal_found.append(replacement)
            text = re.sub(pattern, f" {replacement} ", text)

    # 4. Strip punctuation, keeping alphanumeric
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    # 5. Tokenize
    all_tokens = [t for t in text.split() if t]
    
    # 6. Core tokens (strip stopwords and generic legal words for blocking/similarity)
    core_tokens = [
        t for t in all_tokens 
        if t not in GENERIC_NAME_STOPWORDS and t not in LEGAL_SUFFIX_TOKENS
    ]
    if not core_tokens:
        # Fall back to all tokens if all were filtered
        core_tokens = [t for t in all_tokens if t not in GENERIC_NAME_STOPWORDS] or all_tokens

    # 7. Phonetic keys using Metaphone
    phonetics = []
    for t in core_tokens:
        if t.isalpha():
            code = jellyfish.metaphone(t)
            if code:
                phonetics.append(code)
    phonetic_key = " ".join(phonetics)

    return {
        "name_raw": raw_name,
        "name_norm": text,
        "name_tokens": all_tokens,
        "core_tokens": core_tokens,
        "legal_suffix": " ".join(legal_found),
        "phonetic_key": phonetic_key,
    }
