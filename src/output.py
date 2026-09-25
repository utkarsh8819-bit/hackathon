"""
Output generation for internal audit and competition submissions.
"""
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union
import pandas as pd

from src.config import (
    CANDIDATE_HEADER,
    MATCHING_HEADER,
    INTERNAL_CANDIDATE_HEADER,
)
from src.io import write_tsv


def export_candidate_pairs(
    internal_candidate_df: pd.DataFrame,
    s1_entity_ids: List[str],
    output_dir: Union[str, Path],
    save_internal_4col: bool = True,
) -> Tuple[Path, Optional[Path]]:
    """
    Exports candidate pairs to:
    1. output/candidate_pairs.tsv (competition 2-col format)
    2. output/candidate_pairs_internal.tsv (internal 4-col audit format)
    Ensures every S1 entity is present, empty strings for no candidates,
    no intra-row duplicate IDs, and preserved ID strings.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Internal 4-column table
    internal_path = None
    if save_internal_4col:
        internal_path = output_dir / "candidate_pairs_internal.tsv"
        write_tsv(internal_candidate_df, internal_path, columns=INTERNAL_CANDIDATE_HEADER)

    # 2. Derive competition 2-column format
    if len(internal_candidate_df) > 0:
        grouped = (
            internal_candidate_df.groupby("source1_entity_id")["candidate_entity_id"]
            .apply(lambda ids: ",".join(dict.fromkeys(ids)))  # preserve order, eliminate dupes
            .to_dict()
        )
    else:
        grouped = {}

    comp_rows = []
    for s1_id in s1_entity_ids:
        cands = grouped.get(s1_id, "")
        comp_rows.append({"source1_entity_id": s1_id, "candidate_entity_ids": cands})

    comp_df = pd.DataFrame(comp_rows, columns=CANDIDATE_HEADER)
    comp_path = output_dir / "candidate_pairs.tsv"
    write_tsv(comp_df, comp_path)

    return comp_path, internal_path


def export_matching_results(
    predictions_map: Dict[str, List[str]],
    s1_entity_ids: List[str],
    output_dir: Union[str, Path],
) -> Path:
    """
    Exports matching results to output/matching_results.tsv.
    Follows competition rules:
    - Every S1 entity in the split appears exactly once.
    - Singletons have empty string (not None, not NaN).
    - Comma-separated list with no spaces or quoting.
    - Preserves ID strings.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for s1_id in s1_entity_ids:
        matches = predictions_map.get(s1_id, [])
        # Deduplicate while preserving order
        unique_matches = list(dict.fromkeys(matches))
        # Ensure only S2-/S3- IDs
        filtered_matches = [m for m in unique_matches if m.startswith(("S2-", "S3-"))]
        match_str = ",".join(filtered_matches)
        rows.append({"source1_entity_id": s1_id, "matched_entity_ids": match_str})

    df = pd.DataFrame(rows, columns=MATCHING_HEADER)
    matching_path = output_dir / "matching_results.tsv"
    write_tsv(df, matching_path)
    return matching_path
