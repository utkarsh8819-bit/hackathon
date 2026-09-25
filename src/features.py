"""
Pairwise feature engineering for business entity matching.
"""
from collections import Counter
import math
from typing import Dict, List, Set, Union
import numpy as np
import pandas as pd
import rapidfuzz
from rapidfuzz import fuzz, distance

from src.normalization import normalize_business_name, normalize_country
from src.address_parser import normalize_address


ALL_BLOCKING_RULES = [
    "exact_name",
    "name_token_overlap",
    "phonetic_name",
    "postal_code",
    "address_token_overlap",
    "char_ngram",
]


def token_jaccard(tokens1: List[str], tokens2: List[str]) -> float:
    """Calculates Jaccard similarity between two token lists."""
    s1, s2 = set(tokens1), set(tokens2)
    if not s1 and not s2:
        return 0.0
    intersection = len(s1.intersection(s2))
    union = len(s1.union(s2))
    return float(intersection / union) if union > 0 else 0.0


def char_ngram_cosine(text1: str, text2: str, n: int = 3) -> float:
    """Calculates cosine similarity between character n-gram frequency bags."""
    if not text1 or not text2:
        return 0.0

    def get_counts(s):
        s_clean = "".join(s.split())
        if len(s_clean) < n:
            return Counter([s_clean]) if s_clean else Counter()
        return Counter(s_clean[i : i + n] for i in range(len(s_clean) - n + 1))

    c1, c2 = get_counts(text1), get_counts(text2)
    if not c1 or not c2:
        return 0.0

    dot = sum(c1[g] * c2[g] for g in c1 if g in c2)
    norm1 = math.sqrt(sum(v * v for v in c1.values()))
    norm2 = math.sqrt(sum(v * v for v in c2.values()))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(dot / (norm1 * norm2))


def compute_pair_features(
    s1_record: Dict[str, any],
    s23_record: Dict[str, any],
    blocking_reasons_str: str = "",
) -> Dict[str, float]:
    """
    Computes a comprehensive vector of similarity features for a candidate pair.
    Both records are expected to have:
      business_name, business_address, country (or precomputed normalized representations).
    """
    # Extract or precompute fields
    s1_name_raw = str(s1_record.get("business_name", ""))
    s2_name_raw = str(s23_record.get("business_name", ""))
    s1_addr_raw = str(s1_record.get("business_address", ""))
    s2_addr_raw = str(s23_record.get("business_address", ""))
    s1_country = normalize_country(str(s1_record.get("country", "")))
    s2_country = normalize_country(str(s23_record.get("country", "")))

    s1_name_norm = normalize_business_name(s1_name_raw)
    s2_name_norm = normalize_business_name(s2_name_raw)
    s1_addr_norm = normalize_address(s1_addr_raw)
    s2_addr_norm = normalize_address(s2_addr_raw)

    n1 = s1_name_norm["name_norm"]
    n2 = s2_name_norm["name_norm"]
    a1 = s1_addr_norm["addr_norm"]
    a2 = s2_addr_norm["addr_norm"]

    # 1. Name Similarities
    feat_name_ratio = fuzz.ratio(n1, n2) / 100.0
    feat_name_jw = distance.JaroWinkler.similarity(n1, n2)
    feat_name_token_set = fuzz.token_set_ratio(n1, n2) / 100.0
    feat_name_token_sort = fuzz.token_sort_ratio(n1, n2) / 100.0
    feat_name_ngram_cos = char_ngram_cosine(n1, n2, n=3)

    # Phonetic similarity
    p1 = s1_name_norm["phonetic_key"]
    p2 = s2_name_norm["phonetic_key"]
    feat_phonetic_sim = (fuzz.ratio(p1, p2) / 100.0) if (p1 and p2) else 0.0

    # 2. Address Similarities
    feat_addr_ratio = fuzz.ratio(a1, a2) / 100.0
    feat_addr_jw = distance.JaroWinkler.similarity(a1, a2)
    feat_addr_token_set = fuzz.token_set_ratio(a1, a2) / 100.0
    feat_addr_jaccard = token_jaccard(
        s1_addr_norm["distinctive_tokens"], s2_addr_norm["distinctive_tokens"]
    )
    feat_addr_ngram_cos = char_ngram_cosine(a1, a2, n=3)

    # 3. Structured Subfields (Postal Code, House Number, Country)
    pin1 = s1_addr_norm["postal_code"]
    pin2 = s2_addr_norm["postal_code"]
    if pin1 and pin2:
        feat_same_postal = 1.0 if pin1 == pin2 else 0.0
    else:
        feat_same_postal = -1.0  # missing indicator

    h1 = s1_addr_norm["house_number"]
    h2 = s2_addr_norm["house_number"]
    if h1 and h2:
        feat_same_house = 1.0 if h1 == h2 else 0.0
    else:
        feat_same_house = -1.0  # missing indicator

    # Open-set country equality
    if s1_country and s2_country:
        feat_same_country = 1.0 if s1_country == s2_country else 0.0
    else:
        feat_same_country = 0.5  # unknown

    # 4. Length Discrepancies
    l_n1, l_n2 = len(n1), len(n2)
    feat_name_len_diff_abs = float(abs(l_n1 - l_n2))
    feat_name_len_diff_rel = float(abs(l_n1 - l_n2) / max(l_n1, l_n2, 1))

    l_a1, l_a2 = len(a1), len(a2)
    feat_addr_len_diff_abs = float(abs(l_a1 - l_a2))
    feat_addr_len_diff_rel = float(abs(l_a1 - l_a2) / max(l_a1, l_a2, 1))

    # 5. Multi-hot Blocking Reason Flags
    reasons_set = set(blocking_reasons_str.split(",")) if blocking_reasons_str else set()
    blocking_features = {
        f"block_{rule}": (1.0 if rule in reasons_set else 0.0)
        for rule in ALL_BLOCKING_RULES
    }
    feat_num_blocking_reasons = float(len(reasons_set))

    features = {
        "name_ratio": feat_name_ratio,
        "name_jaro_winkler": feat_name_jw,
        "name_token_set": feat_name_token_set,
        "name_token_sort": feat_name_token_sort,
        "name_ngram_cos": feat_name_ngram_cos,
        "phonetic_sim": feat_phonetic_sim,
        "addr_ratio": feat_addr_ratio,
        "addr_jaro_winkler": feat_addr_jw,
        "addr_token_set": feat_addr_token_set,
        "addr_jaccard": feat_addr_jaccard,
        "addr_ngram_cos": feat_addr_ngram_cos,
        "same_postal": feat_same_postal,
        "same_house": feat_same_house,
        "same_country": feat_same_country,
        "name_len_diff_abs": feat_name_len_diff_abs,
        "name_len_diff_rel": feat_name_len_diff_rel,
        "addr_len_diff_abs": feat_addr_len_diff_abs,
        "addr_len_diff_rel": feat_addr_len_diff_rel,
        "num_blocking_reasons": feat_num_blocking_reasons,
        **blocking_features,
    }

    return features


def extract_features_for_pairs(
    candidate_pairs_df: pd.DataFrame,
    s1_dict: Dict[str, Dict[str, any]],
    s23_dict: Dict[str, Dict[str, any]],
) -> pd.DataFrame:
    """
    Computes pairwise feature matrix for a dataframe containing
    [source1_entity_id, candidate_entity_id, blocking_reasons (optional)].
    """
    feature_rows = []
    
    for _, row in candidate_pairs_df.iterrows():
        s1_id = str(row["source1_entity_id"]).strip()
        cand_id = str(row["candidate_entity_id"]).strip()
        reasons = str(row.get("blocking_reasons", ""))

        s1_rec = s1_dict.get(s1_id, {})
        s23_rec = s23_dict.get(cand_id, {})

        feats = compute_pair_features(s1_rec, s23_rec, blocking_reasons_str=reasons)
        feats["source1_entity_id"] = s1_id
        feats["candidate_entity_id"] = cand_id
        feature_rows.append(feats)

    return pd.DataFrame(feature_rows)
