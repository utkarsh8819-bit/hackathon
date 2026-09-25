"""
Address normalization and structured sub-field extraction module.
"""
import re
from typing import Dict, List, Optional
from src.normalization import unicode_clean

# Common address abbreviation expansions
ADDRESS_ABBREV_MAP = {
    r"\brd\.?\b": "road",
    r"\bst\.?\b": "street",
    r"\bave\.?\b": "avenue",
    r"\bblvd\.?\b": "boulevard",
    r"\bdr\.?\b": "drive",
    r"\bln\.?\b": "lane",
    r"\bhwy\.?\b": "highway",
    r"\bste\.?\b": "suite",
    r"\bapt\.?\b": "apartment",
    r"\bbldg\.?\b": "building",
    r"\bflr?\.?\b": "floor",
    r"\bindl\.?\b": "industrial",
    r"\bind\.?\b": "industrial",
    r"\bopp\.?\b": "opposite",
    r"\bnr\.?\b": "near",
    r"\bpk\.?\b": "park",
    r"\bct\.?\b": "court",
    r"\bsq\.?\b": "square",
    r"\bctr\.?\b": "center",
    r"\bdept\.?\b": "department",
    r"\bmt\.?\b": "mount",
    r"\bn\.\b": "north",
    r"\bs\.\b": "south",
    r"\be\.\b": "east",
    r"\bw\.\b": "west",
}

GENERIC_ADDR_TOKENS = {
    "road", "street", "avenue", "drive", "lane", "suite", "floor",
    "building", "block", "plot", "near", "opposite", "area", "phase",
    "sector", "nagar", "marg", "cross", "main", "east", "west",
    "north", "south", "in", "at", "to", "and", "of", "the", "rue", "de", "la"
}


def extract_postal_code(raw_addr: str) -> str:
    """
    Extracts postal code (6-digit Indian PIN, 5-digit US Zip or French code).
    Avoids matching generic short numbers.
    """
    # 6-digit Indian PIN (often starts with 1-8, but any 6-digit isolated number)
    match_6 = re.findall(r"\b[1-9]\d{5}\b", raw_addr)
    if match_6:
        return match_6[-1]

    # 5-digit US Zip code or French postal code
    match_5 = re.findall(r"\b\d{5}\b", raw_addr)
    if match_5:
        return match_5[-1]

    return ""


def extract_house_or_building(raw_addr: str) -> str:
    """
    Extracts building / house / plot / block numbers.
    Prioritizes leading street/house numbers, followed by plot/block designations.
    """
    # Leading number pattern: e.g., '1200 Market', '12/4 Gandhi Road', '15 Rue'
    lead_match = re.search(r"^\s*([0-9]+[a-z]?(?:/[0-9]+)?)\b", raw_addr, re.IGNORECASE)
    if lead_match:
        return lead_match.group(1).lower()

    # Plot or Block pattern: e.g., 'plot 45', 'block a', 'shop 12'
    plot_match = re.search(r"\b(plot|block|shop|flat|suite|ste|no\.?)\s*([a-z0-9/-]+)\b", raw_addr, re.IGNORECASE)
    if plot_match:
        return f"{plot_match.group(1).lower()} {plot_match.group(2).lower()}"

    # Comma-delimited number token: e.g., ', 797, Lake Town'
    comma_num = re.search(r",\s*([0-9]+[a-z]?)\s*,", raw_addr)
    if comma_num:
        return comma_num.group(1).lower()

    return ""


def extract_landmark(raw_addr: str) -> str:
    """
    Extracts landmark descriptions common in Indian addresses (e.g. 'Near SBI ATM').
    """
    match = re.search(r"\b(near|opp|opposite|behind|beside)\s+([a-zA-Z0-9\s]{3,35}?)(?:,|$|\b(?:road|street|nagar|marg)\b)", raw_addr, re.IGNORECASE)
    if match:
        return re.sub(r"\s+", " ", match.group(0).lower()).strip()
    return ""


def normalize_address(raw_addr: str) -> Dict[str, any]:
    """
    Normalizes address text, standardizes street abbreviations, extracts
    postal code, building number, landmark, and distinctive tokens.
    """
    if not raw_addr:
        return {
            "addr_raw": "",
            "addr_norm": "",
            "addr_tokens": [],
            "postal_code": "",
            "house_number": "",
            "landmark": "",
            "distinctive_tokens": [],
        }

    # Extract structured fields before heavy stripping
    postal_code = extract_postal_code(raw_addr)
    house_number = extract_house_or_building(raw_addr)
    landmark = extract_landmark(raw_addr)

    # Unicode & lowercase
    text = unicode_clean(raw_addr).lower()

    # Expand abbreviations
    for pattern, replacement in ADDRESS_ABBREV_MAP.items():
        text = re.sub(pattern, f" {replacement} ", text)

    # Strip punctuation
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    all_tokens = [t for t in text.split() if t]
    distinctive_tokens = [
        t for t in all_tokens
        if t not in GENERIC_ADDR_TOKENS and not (t.isdigit() and len(t) < 3)
    ]

    return {
        "addr_raw": raw_addr,
        "addr_norm": text,
        "addr_tokens": all_tokens,
        "postal_code": postal_code,
        "house_number": house_number,
        "landmark": landmark,
        "distinctive_tokens": distinctive_tokens,
    }
