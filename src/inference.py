"""
Inference pipeline for test split and submission generation.
"""
import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.config import (
    OUTPUT_DIR,
    MODELS_DIR,
    MODEL_FILE,
    THRESHOLD_FILE,
    MATCHING_RESULTS_FILE,
    CANDIDATE_PAIRS_FILE,
)
from src.io import load_split_datasets
from src.blocking import BlockingEngine
from src.features import extract_features_for_pairs
from src.train import FEATURE_COLS
from src.output import export_candidate_pairs, export_matching_results


def run_inference_pipeline(
    test_dir: Optional[Union[str, Path]] = None,
    test_dict: Optional[Dict[str, any]] = None,
    output_dir: Union[str, Path] = OUTPUT_DIR,
    models_dir: Union[str, Path] = MODELS_DIR,
    override_threshold: Optional[float] = None,
) -> Tuple[Path, Path]:
    """
    Executes complete inference:
    1. Loads test sources (S1, S2, S3)
    2. Runs BlockingEngine to generate candidates for all S1 test entities
    3. Exports output/candidate_pairs.tsv and candidate_pairs_internal.tsv
    4. Computes pairwise features
    5. Scores pairs with trained ML matcher
    6. Filters matches using optimal validation threshold
    7. Exports output/matching_results.tsv
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir = Path(models_dir)

    print("=== Step 1: Loading Test Data ===")
    if test_dict is None:
        if test_dir is None:
            raise ValueError("Must provide test_dir or test_dict")
        test_dict = load_split_datasets(test_dir, is_train=False)

    s1_df = test_dict["source1"]
    s2_df = test_dict["source2"]
    s3_df = test_dict["source3"]
    all_s1_ids = s1_df["entity_id"].tolist()
    print(f"Loaded test split: S1={len(s1_df)}, S2={len(s2_df)}, S3={len(s3_df)}")

    # Verify open-set countries
    countries_found = s1_df["country"].unique().tolist()
    print(f"Test S1 countries present (open-set verified): {countries_found}")

    print("=== Step 2: Running Blocking on Candidate Pool (S2 + S3) ===")
    blocking_engine = BlockingEngine()
    blocking_engine.fit(s2_df, s3_df)
    internal_cand_df, _ = blocking_engine.generate_candidates(s1_df)
    print(f"Generated {len(internal_cand_df)} total candidate pairs for {len(all_s1_ids)} S1 entities")

    print("=== Step 3: Exporting candidate_pairs.tsv ===")
    cand_path, int_cand_path = export_candidate_pairs(
        internal_cand_df,
        s1_entity_ids=all_s1_ids,
        output_dir=output_dir,
        save_internal_4col=True,
    )
    print(f"Exported candidate pairs to: {cand_path}")

    print("=== Step 4: Loading Model & Optimal Threshold ===")
    model_path = models_dir / MODEL_FILE
    threshold_path = models_dir / THRESHOLD_FILE

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with open(threshold_path, "r", encoding="utf-8") as f:
        thresh_meta = json.load(f)

    threshold = (
        override_threshold if override_threshold is not None else float(thresh_meta["best_threshold"])
    )
    print(f"Loaded model ({thresh_meta.get('model_name', 'Model')}) with threshold: {threshold:.4f}")

    print("=== Step 5: Scoring Candidate Pairs ===")
    predictions_map: Dict[str, List[str]] = {s1: [] for s1 in all_s1_ids}

    if len(internal_cand_df) > 0:
        s1_dict = s1_df.set_index("entity_id").to_dict(orient="index")
        s23_combined = pd.concat([s2_df, s3_df], ignore_index=True)
        s23_dict = s23_combined.set_index("entity_id").to_dict(orient="index")

        X_cand_df = extract_features_for_pairs(internal_cand_df, s1_dict, s23_dict)
        X_test = X_cand_df[FEATURE_COLS].values

        probs = model.predict_proba(X_test)[:, 1]
        internal_cand_df["prob"] = probs

        # Filter accepted matches
        accepted_df = internal_cand_df[internal_cand_df["prob"] >= threshold]
        
        # Sort by probability descending within each S1
        accepted_sorted = accepted_df.sort_values(by=["source1_entity_id", "prob"], ascending=[True, False])
        
        grouped = accepted_sorted.groupby("source1_entity_id")["candidate_entity_id"].apply(list)
        for s1_id, matches in grouped.items():
            predictions_map[s1_id] = matches

    print("=== Step 6: Exporting matching_results.tsv ===")
    match_path = export_matching_results(
        predictions_map,
        s1_entity_ids=all_s1_ids,
        output_dir=output_dir,
    )
    print(f"Exported matching results to: {match_path}")

    non_empty_count = sum(1 for m in predictions_map.values() if len(m) > 0)
    print(f"Summary: {len(all_s1_ids)} S1 entities -> {non_empty_count} matched, "
          f"{len(all_s1_ids) - non_empty_count} singletons")

    return match_path, cand_path


def main():
    parser = argparse.ArgumentParser(description="Run inference on test set and output matching results")
    parser.add_argument("--test-dir", default="dataset/test", help="Path to test TSVs directory")
    parser.add_argument("--output-dir", default="output", help="Directory to save TSV outputs")
    parser.add_argument("--models-dir", default="models", help="Directory where matcher.pkl is saved")
    parser.add_argument("--threshold", type=float, default=None, help="Optional override for matching threshold")
    args = parser.parse_args()

    run_inference_pipeline(
        test_dir=args.test_dir,
        output_dir=args.output_dir,
        models_dir=args.models_dir,
        override_threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
