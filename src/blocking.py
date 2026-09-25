"""
Multi-rule inverted-index blocking engine for candidate pair generation.
"""
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple
import pandas as pd

from src.config import (
    INTERNAL_CANDIDATE_HEADER,
    CANDIDATE_HEADER,
    MAX_CANDIDATES_PER_S1,
)
from src.normalization import normalize_business_name, normalize_country
from src.address_parser import normalize_address


def get_char_ngrams(text: str, n: int = 3) -> Set[str]:
    """Generates character n-grams from normalized text."""
    clean = "".join(text.split())
    if len(clean) < n:
        return {clean} if clean else set()
    return {clean[i : i + n] for i in range(len(clean) - n + 1)}


class BlockingEngine:
    """
    High-recall blocking engine using multiple inverted indexes.
    Indexed over S2 and S3 candidate entities.
    """

    def __init__(
        self,
        max_candidates_per_query: int = MAX_CANDIDATES_PER_S1,
        max_token_postings: int = 1000,
    ):
        self.max_candidates_per_query = max_candidates_per_query
        self.max_token_postings = max_token_postings

        # Inverted indexes: token -> list of candidate entity_ids
        self.exact_name_index = defaultdict(list)
        self.name_token_index = defaultdict(list)
        self.phonetic_index = defaultdict(list)
        self.postal_index = defaultdict(list)
        self.addr_token_index = defaultdict(list)
        self.ngram_index = defaultdict(list)

        # Storage for candidate metadata
        self.candidates_meta: Dict[str, Dict[str, any]] = {}

    def fit(self, s2_df: pd.DataFrame, s3_df: pd.DataFrame) -> "BlockingEngine":
        """Builds inverted indexes across all S2 and S3 records."""
        combined_pool = pd.concat([s2_df, s3_df], ignore_index=True)

        for _, row in combined_pool.iterrows():
            cid = str(row["entity_id"]).strip()
            raw_name = str(row["business_name"])
            raw_addr = str(row["business_address"])
            raw_country = str(row["country"])

            norm_name = normalize_business_name(raw_name)
            norm_addr = normalize_address(raw_addr)
            country = normalize_country(raw_country)

            self.candidates_meta[cid] = {
                "entity_id": cid,
                "source": "S2" if cid.startswith("S2-") else "S3",
                "name_norm": norm_name["name_norm"],
                "core_tokens": norm_name["core_tokens"],
                "phonetic_key": norm_name["phonetic_key"],
                "postal_code": norm_addr["postal_code"],
                "addr_tokens": norm_addr["distinctive_tokens"],
                "country": country,
            }

            # 1. Exact normalized name
            if norm_name["name_norm"]:
                self.exact_name_index[norm_name["name_norm"]].append(cid)

            # 2. Informative name tokens
            for token in norm_name["core_tokens"]:
                if len(token) >= 3:
                    self.name_token_index[token].append(cid)

            # 3. Phonetic tokens
            if norm_name["phonetic_key"]:
                for p_code in norm_name["phonetic_key"].split():
                    if len(p_code) >= 2:
                        self.phonetic_index[p_code].append(cid)

            # 4. Postal code (when present)
            if norm_addr["postal_code"]:
                self.postal_index[norm_addr["postal_code"]].append(cid)

            # 5. Distinctive address tokens
            for atoken in norm_addr["distinctive_tokens"]:
                if len(atoken) >= 4:
                    self.addr_token_index[atoken].append(cid)

            # 6. Character 3-grams of name
            for gram in get_char_ngrams(norm_name["name_norm"], n=3):
                self.ngram_index[gram].append(cid)

        return self

    def query_entity(self, s1_row: pd.Series) -> Dict[str, Set[str]]:
        """
        Retrieves candidate entity_ids and tracks blocking reasons.
        Returns: {candidate_id: {reason1, reason2, ...}}
        """
        raw_name = str(s1_row["business_name"])
        raw_addr = str(s1_row["business_address"])
        raw_country = str(s1_row.get("country", ""))

        norm_name = normalize_business_name(raw_name)
        norm_addr = normalize_address(raw_addr)
        country = normalize_country(raw_country)

        candidates_reasons: Dict[str, Set[str]] = defaultdict(set)

        # Rule 1: Exact Name
        if norm_name["name_norm"] and norm_name["name_norm"] in self.exact_name_index:
            for cid in self.exact_name_index[norm_name["name_norm"]]:
                candidates_reasons[cid].add("exact_name")

        # Rule 2: Informative Name Token Overlap
        token_hits = defaultdict(int)
        for token in norm_name["core_tokens"]:
            if len(token) >= 3 and token in self.name_token_index:
                postings = self.name_token_index[token]
                if len(postings) <= self.max_token_postings:
                    for cid in postings:
                        token_hits[cid] += 1

        min_token_overlap = 1 if len(norm_name["core_tokens"]) <= 2 else 2
        for cid, hits in token_hits.items():
            if hits >= min_token_overlap:
                candidates_reasons[cid].add("name_token_overlap")

        # Rule 3: Phonetic Matching
        if norm_name["phonetic_key"]:
            phon_hits = defaultdict(int)
            for p_code in norm_name["phonetic_key"].split():
                if len(p_code) >= 2 and p_code in self.phonetic_index:
                    postings = self.phonetic_index[p_code]
                    if len(postings) <= self.max_token_postings:
                        for cid in postings:
                            phon_hits[cid] += 1
            for cid, hits in phon_hits.items():
                if hits >= 1:
                    candidates_reasons[cid].add("phonetic_name")

        # Rule 4: Postal Code match
        if norm_addr["postal_code"] and norm_addr["postal_code"] in self.postal_index:
            for cid in self.postal_index[norm_addr["postal_code"]]:
                # Require some token or country agreement to avoid pure zip code flood
                cand_meta = self.candidates_meta.get(cid, {})
                if cand_meta.get("country") == country or not country:
                    candidates_reasons[cid].add("postal_code")

        # Rule 5: Address Token Overlap
        addr_hits = defaultdict(int)
        for atoken in norm_addr["distinctive_tokens"]:
            if len(atoken) >= 4 and atoken in self.addr_token_index:
                postings = self.addr_token_index[atoken]
                if len(postings) <= self.max_token_postings:
                    for cid in postings:
                        addr_hits[cid] += 1
        for cid, hits in addr_hits.items():
            if hits >= 2:
                candidates_reasons[cid].add("address_token_overlap")

        # Rule 6: Name Character 3-Gram Overlap
        s1_grams = get_char_ngrams(norm_name["name_norm"], n=3)
        if len(s1_grams) >= 3:
            gram_hits = defaultdict(int)
            for g in s1_grams:
                if g in self.ngram_index:
                    postings = self.ngram_index[g]
                    if len(postings) <= self.max_token_postings:
                        for cid in postings:
                            gram_hits[cid] += 1
            threshold_hits = max(3, int(len(s1_grams) * 0.4))
            for cid, hits in gram_hits.items():
                if hits >= threshold_hits:
                    candidates_reasons[cid].add("char_ngram")

        # Prioritize candidates and cap per query
        if len(candidates_reasons) > self.max_candidates_per_query:
            # Sort by number of independent blocking reasons descending
            sorted_candidates = sorted(
                candidates_reasons.items(),
                key=lambda item: len(item[1]),
                reverse=True,
            )
            candidates_reasons = dict(sorted_candidates[: self.max_candidates_per_query])

        return candidates_reasons

    def generate_candidates(
        self, s1_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generates candidates for every S1 entity in s1_df.
        Returns:
            - internal_df: 4 columns (source1_entity_id, candidate_entity_id, candidate_source, blocking_reasons)
            - competition_df: 2 columns (source1_entity_id, candidate_entity_ids)
        """
        internal_rows = []
        comp_rows = []

        for _, s1_row in s1_df.iterrows():
            s1_id = str(s1_row["entity_id"]).strip()
            cands_dict = self.query_entity(s1_row)

            cand_id_list = []
            for cid, reasons in cands_dict.items():
                cand_id_list.append(cid)
                source_tag = "S2" if cid.startswith("S2-") else "S3"
                internal_rows.append({
                    "source1_entity_id": s1_id,
                    "candidate_entity_id": cid,
                    "candidate_source": source_tag,
                    "blocking_reasons": ",".join(sorted(reasons)),
                })

            comp_rows.append({
                "source1_entity_id": s1_id,
                "candidate_entity_ids": ",".join(cand_id_list),
            })

        internal_df = pd.DataFrame(internal_rows, columns=INTERNAL_CANDIDATE_HEADER)
        competition_df = pd.DataFrame(comp_rows, columns=CANDIDATE_HEADER)

        return internal_df, competition_df
