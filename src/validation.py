"""
Schema and data validation for Entity Resolution pipeline.
"""
from typing import Dict, List, Optional, Set, Tuple
import pandas as pd

from src.config import SOURCE_COLUMNS, GROUND_TRUTH_COLUMNS


class ValidationError(Exception):
    """Custom exception raised when data contract is violated."""
    pass


def validate_source_dataframe(
    df: pd.DataFrame,
    expected_prefix: str,
    dataset_name: str = "source",
    strict: bool = True,
) -> Dict[str, any]:
    """
    Validates a source DataFrame against the data contract:
    1. Required columns exist.
    2. entity_id uniqueness.
    3. Correct ID prefix (S1-, S2-, S3-).
    4. Reports null rates and duplicate rows.
    """
    issues = []
    
    # Check columns
    missing_cols = [c for c in SOURCE_COLUMNS if c not in df.columns]
    if missing_cols:
        issues.append(f"Missing required columns in {dataset_name}: {missing_cols}")

    if "entity_id" in df.columns:
        # Check uniqueness
        dup_ids = df[df["entity_id"].duplicated()]["entity_id"].tolist()
        if dup_ids:
            issues.append(
                f"Duplicate entity_id found in {dataset_name} ({len(dup_ids)} duplicates, e.g. {dup_ids[:3]})"
            )

        # Check prefix
        invalid_prefix_mask = ~df["entity_id"].astype(str).str.startswith(expected_prefix)
        invalid_prefix_count = invalid_prefix_mask.sum()
        if invalid_prefix_count > 0:
            sample_invalid = df.loc[invalid_prefix_mask, "entity_id"].head(3).tolist()
            issues.append(
                f"Found {invalid_prefix_count} entity_ids not starting with '{expected_prefix}' in {dataset_name}: {sample_invalid}"
            )

    # Check null rates
    null_counts = df[SOURCE_COLUMNS].isna().sum().to_dict()
    total_rows = len(df)
    
    diagnostics = {
        "dataset_name": dataset_name,
        "total_rows": total_rows,
        "null_counts": null_counts,
        "unique_entities": df["entity_id"].nunique() if "entity_id" in df.columns else 0,
        "countries": df["country"].unique().tolist() if "country" in df.columns else [],
        "issues": issues,
    }

    if issues and strict:
        raise ValidationError("; ".join(issues))

    return diagnostics


def parse_ground_truth(
    gt_df: pd.DataFrame,
    valid_s1_ids: Optional[Set[str]] = None,
    valid_s2_s3_ids: Optional[Set[str]] = None,
    strict: bool = True,
) -> Tuple[Dict[str, List[str]], Dict[str, any]]:
    """
    Parses and validates ground truth:
    - Returns mapping: source1_entity_id -> list of matched S2-/S3- entity_ids
    - Empty string -> empty list (singleton)
    - Validates no duplicate S1 rows, no duplicate IDs in match list, no S1 self-matches.
    - Gracefully handles unknown S2/S3 IDs without crashing: records them as warnings
      and filters them so downstream processing remains stable.
    """
    issues = []
    warnings = []
    
    missing_cols = [c for c in GROUND_TRUTH_COLUMNS if c not in gt_df.columns]
    if missing_cols:
        raise ValidationError(f"Ground truth missing required columns: {missing_cols}")

    s1_col = gt_df["source1_entity_id"].astype(str)
    dup_s1 = gt_df[s1_col.duplicated()]["source1_entity_id"].tolist()
    if dup_s1:
        issues.append(f"Duplicate source1_entity_id rows in ground truth: {dup_s1[:5]}")

    # Initialize with all valid S1 entities if provided
    gt_map: Dict[str, List[str]] = {s1: [] for s1 in valid_s1_ids} if valid_s1_ids else {}
    
    total_raw_matches = 0
    total_valid_matches = 0
    singletons = 0
    self_matches = []
    unknown_ids = []

    for _, row in gt_df.iterrows():
        s1_id = str(row["source1_entity_id"]).strip()
        
        # If valid_s1_ids is restricted and s1_id is not in it, skip
        if valid_s1_ids is not None and s1_id not in valid_s1_ids:
            continue

        raw_matches = str(row["matched_entity_ids"]).strip() if pd.notna(row["matched_entity_ids"]) else ""

        if not raw_matches or raw_matches.lower() in ("none", "nan"):
            matched_list = []
            singletons += 1
            gt_map[s1_id] = []
        else:
            matched_list = [m.strip() for m in raw_matches.split(",") if m.strip()]
            
            # Check for intra-row duplicates
            if len(matched_list) != len(set(matched_list)):
                warnings.append(f"Intra-row duplicate IDs for S1 ID '{s1_id}': {matched_list}")
                matched_list = list(dict.fromkeys(matched_list))
            
            valid_matches_for_s1 = []
            for mid in matched_list:
                total_raw_matches += 1
                if mid.startswith("S1-"):
                    self_matches.append(mid)
                elif not mid.startswith(("S2-", "S3-")):
                    issues.append(f"Invalid match ID prefix: '{mid}' for '{s1_id}'")
                elif valid_s2_s3_ids is not None and mid not in valid_s2_s3_ids:
                    unknown_ids.append(mid)
                else:
                    valid_matches_for_s1.append(mid)

            total_valid_matches += len(valid_matches_for_s1)
            gt_map[s1_id] = valid_matches_for_s1

    if self_matches:
        issues.append(f"Ground truth contains Source 1 self-matches: {self_matches[:5]}")

    if unknown_ids:
        warnings.append(
            f"{len(unknown_ids)} ground truth ID(s) not found in S2/S3 pool (e.g. {unknown_ids[:5]}). "
            "These have been filtered out of active matching."
        )

    stats = {
        "total_s1_entities": len(gt_map),
        "total_raw_matches": total_raw_matches,
        "total_valid_matches": total_valid_matches,
        "unknown_ids_count": len(unknown_ids),
        "singletons_count": sum(1 for v in gt_map.values() if len(v) == 0),
        "multi_match_count": sum(1 for v in gt_map.values() if len(v) > 1),
        "issues": issues,
        "warnings": warnings,
    }

    if warnings:
        for w in warnings:
            print(f"VALIDATION WARNING: {w}")

    if issues and strict:
        raise ValidationError("; ".join(issues))

    return gt_map, stats
