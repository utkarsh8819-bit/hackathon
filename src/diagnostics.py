"""
Failure analysis and error bucketing diagnostic tools.
"""
from typing import Dict, List, Set, Tuple
import pandas as pd


def analyze_errors(
    val_pairs_df: pd.DataFrame,
    val_probs: pd.Series,
    gt_map: Dict[str, List[str]],
    val_s1_ids: List[str],
    threshold: float,
) -> Dict[str, any]:
    """
    Buckets validation errors into the 4 challenge categories:
    1. Blocking False Negatives: True match never appeared in candidate_pairs.
    2. Scoring False Negatives: True candidate retrieved by blocking but score < threshold.
    3. Scoring False Positives: Non-match candidate scored >= threshold (false merge).
    4. Threshold Borderline: Candidates with |score - threshold| <= 0.05.
    """
    df = val_pairs_df.copy()
    df["score"] = val_probs

    # Candidates grouped by s1
    candidates_by_s1 = df.groupby("source1_entity_id")["candidate_entity_id"].apply(set).to_dict()

    blocking_fn = []
    scoring_fn = []
    scoring_fp = []
    borderline = []

    # Category 1: Blocking False Negatives
    for s1_id in val_s1_ids:
        true_matches = set(gt_map.get(s1_id, []))
        cands = candidates_by_s1.get(s1_id, set())
        for tm in true_matches:
            if tm not in cands:
                blocking_fn.append({
                    "source1_entity_id": s1_id,
                    "target_entity_id": tm,
                    "error_type": "blocking_false_negative",
                    "reason": "True match not retrieved by any blocking rule",
                })

    # Category 2, 3, 4: Scoring errors & threshold borderlines
    for _, row in df.iterrows():
        s1_id = row["source1_entity_id"]
        cand_id = row["candidate_entity_id"]
        score = row["score"]
        is_true_match = cand_id in set(gt_map.get(s1_id, []))

        # Check borderline
        if abs(score - threshold) <= 0.05:
            borderline.append({
                "source1_entity_id": s1_id,
                "candidate_entity_id": cand_id,
                "score": score,
                "is_true_match": is_true_match,
                "threshold": threshold,
            })

        if is_true_match and score < threshold:
            # Scoring False Negative
            scoring_fn.append({
                "source1_entity_id": s1_id,
                "candidate_entity_id": cand_id,
                "score": score,
                "threshold": threshold,
                "error_type": "scoring_false_negative",
                "blocking_reasons": row.get("blocking_reasons", ""),
            })
        elif not is_true_match and score >= threshold:
            # Scoring False Positive
            scoring_fp.append({
                "source1_entity_id": s1_id,
                "candidate_entity_id": cand_id,
                "score": score,
                "threshold": threshold,
                "error_type": "scoring_false_positive",
                "blocking_reasons": row.get("blocking_reasons", ""),
            })

    report = {
        "num_blocking_false_negatives": len(blocking_fn),
        "num_scoring_false_negatives": len(scoring_fn),
        "num_scoring_false_positives": len(scoring_fp),
        "num_borderline_cases": len(borderline),
        "blocking_fn_list": blocking_fn,
        "scoring_fn_list": scoring_fn,
        "scoring_fp_list": scoring_fp,
        "borderline_list": borderline,
    }

    return report


def print_diagnostic_report(report: Dict[str, any]) -> None:
    """Prints a formatted summary of failure analysis."""
    print("=" * 60)
    print("FAILURE ANALYSIS & ERROR BUCKETING REPORT")
    print("=" * 60)
    print(f"1. Blocking False Negatives (never reached candidates): {report['num_blocking_false_negatives']}")
    print(f"2. Scoring False Negatives  (score < threshold)      : {report['num_scoring_false_negatives']}")
    print(f"3. Scoring False Positives  (false merges, score >= T) : {report['num_scoring_fp_list'] if 'num_scoring_fp_list' in report else report['num_scoring_false_positives']}")
    print(f"4. Borderline Threshold Cases (|score - T| <= 0.05)   : {report['num_borderline_cases']}")
    print("=" * 60)
