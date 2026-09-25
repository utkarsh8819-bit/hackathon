"""
Model training and threshold optimization pipeline.
"""
import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier

from src.config import (
    MODELS_DIR,
    MODEL_FILE,
    THRESHOLD_FILE,
    RANDOM_STATE,
    F_BETA,
)
from src.io import load_split_datasets
from src.blocking import BlockingEngine
from src.pair_generation import build_labeled_training_pairs, split_grouped_train_val
from src.features import extract_features_for_pairs
from src.evaluate import (
    compute_pair_level_metrics,
    sweep_decision_threshold,
    compute_macro_f_beta,
)


FEATURE_COLS = [
    "name_ratio",
    "name_jaro_winkler",
    "name_token_set",
    "name_token_sort",
    "name_ngram_cos",
    "phonetic_sim",
    "addr_ratio",
    "addr_jaro_winkler",
    "addr_token_set",
    "addr_jaccard",
    "addr_ngram_cos",
    "same_postal",
    "same_house",
    "same_country",
    "name_len_diff_abs",
    "name_len_diff_rel",
    "addr_len_diff_abs",
    "addr_len_diff_rel",
    "num_blocking_reasons",
    "block_exact_name",
    "block_name_token_overlap",
    "block_phonetic_name",
    "block_postal_code",
    "block_address_token_overlap",
    "block_char_ngram",
]


def train_entity_matching_system(
    data_dir: Optional[Union[str, Path]] = None,
    data_dict: Optional[Dict[str, any]] = None,
    models_dir: Union[str, Path] = MODELS_DIR,
    val_size: float = 0.25,
    random_state: int = RANDOM_STATE,
) -> Dict[str, any]:
    """
    End-to-end training pipeline:
    1. Load data & ground truth.
    2. Build inverted index blocking on S2+S3 and retrieve S1 candidates.
    3. Measure & log blocking recall ceiling.
    4. Generate labeled pairs with hard negatives.
    5. Grouped train/val split (zero leakage).
    6. Extract pairwise features.
    7. Train baseline (Logistic Regression) & main model (LightGBM).
    8. Sweep validation threshold maximizing per-entity macro F_0.5.
    9. Save model and threshold artifacts.
    """
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    print("=== Step 1: Loading Training Data ===")
    if data_dict is None:
        if data_dir is None:
            raise ValueError("Must provide either data_dir or data_dict")
        data_dict = load_split_datasets(data_dir, is_train=True)

    s1_df = data_dict.get("source1", data_dict.get("train_source1"))
    s2_df = data_dict.get("source2", data_dict.get("train_source2"))
    s3_df = data_dict.get("source3", data_dict.get("train_source3"))
    gt_map = data_dict.get("ground_truth_map", {})

    print(f"Loaded: S1={len(s1_df)}, S2={len(s2_df)}, S3={len(s3_df)}, GT={len(gt_map)}")

    print("=== Step 2: Blocking & Candidate Generation ===")
    blocking_engine = BlockingEngine()
    blocking_engine.fit(s2_df, s3_df)
    internal_cand_df, _ = blocking_engine.generate_candidates(s1_df)

    print("=== Step 3: Labeled Pair Construction & Blocking Recall Ceiling ===")
    labeled_pairs_df, pair_stats = build_labeled_training_pairs(
        internal_cand_df, gt_map, random_state=random_state
    )
    print(f"Blocking Candidate Recall Ceiling: {pair_stats['blocking_recall_ceiling']:.4f} "
          f"({pair_stats['survived_positives']}/{pair_stats['total_gt_positives']} positives survived)")
    print(f"Total labeled pairs: {pair_stats['total_labeled_pairs']} "
          f"(Pos: {pair_stats['positive_count']}, Neg: {pair_stats['negative_count']})")

    if pair_stats["positive_count"] == 0:
        raise ValueError(
            "No positive training pairs found in candidate pool! "
            "None of the ground truth match IDs exist in the loaded S2/S3 files. "
            "If using a slice or sample of the data, ensure the sample is connected."
        )

    print("=== Step 4: Grouped Train/Validation Split ===")
    train_pairs_df, val_pairs_df = split_grouped_train_val(
        labeled_pairs_df, val_size=val_size, random_state=random_state
    )
    val_s1_ids = sorted(list(set(val_pairs_df["source1_entity_id"])))
    print(f"Train pairs: {len(train_pairs_df)}, Val pairs: {len(val_pairs_df)} "
          f"across {len(val_s1_ids)} validation S1 entities")

    print("=== Step 5: Pairwise Feature Engineering ===")
    s1_dict = s1_df.set_index("entity_id").to_dict(orient="index")
    s23_combined = pd.concat([s2_df, s3_df], ignore_index=True)
    s23_dict = s23_combined.set_index("entity_id").to_dict(orient="index")

    X_train_df = extract_features_for_pairs(train_pairs_df, s1_dict, s23_dict)
    X_val_df = extract_features_for_pairs(val_pairs_df, s1_dict, s23_dict)

    X_train = X_train_df[FEATURE_COLS].values
    y_train = train_pairs_df["label"].values
    X_val = X_val_df[FEATURE_COLS].values
    y_val = val_pairs_df["label"].values

    print("=== Step 6: Training Models ===")
    # 6.1 Baseline: Logistic Regression
    lr_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(class_weight="balanced", random_state=random_state, max_iter=1000)),
    ])
    lr_pipeline.fit(X_train, y_train)
    val_lr_probs = lr_pipeline.predict_proba(X_val)[:, 1]
    lr_metrics = compute_pair_level_metrics(y_val, val_lr_probs)
    print(f"Baseline (Logistic Regression) Pair Metrics: {lr_metrics}")

    # 6.2 Main Model: LightGBM
    lgb_model = LGBMClassifier(
        n_estimators=120,
        learning_rate=0.05,
        num_leaves=31,
        random_state=random_state,
        class_weight="balanced",
        verbose=-1,
    )
    lgb_model.fit(X_train, y_train)
    val_lgb_probs = lgb_model.predict_proba(X_val)[:, 1]
    lgb_metrics = compute_pair_level_metrics(y_val, val_lgb_probs)
    print(f"Main Model (LightGBM) Pair Metrics: {lgb_metrics}")

    chosen_model = lgb_model
    chosen_probs = val_lgb_probs
    model_name = "LightGBM"

    print("=== Step 7: Threshold Sweep for Competition Metric (Per-Entity Macro F_0.5) ===")
    best_thresh, best_macro_f05, sweep_results = sweep_decision_threshold(
        val_pairs_df, chosen_probs, gt_map, val_s1_ids
    )
    print(f"Optimal Threshold: {best_thresh:.4f} -> Best Validation Macro F_0.5: {best_macro_f05:.4f}")

    print("=== Step 8: Saving Artifacts ===")
    model_path = models_dir / MODEL_FILE
    with open(model_path, "wb") as f:
        pickle.dump(chosen_model, f)

    threshold_info = {
        "model_name": model_name,
        "best_threshold": float(best_thresh),
        "val_macro_f05": float(best_macro_f05),
        "blocking_recall_ceiling": float(pair_stats["blocking_recall_ceiling"]),
        "feature_names": FEATURE_COLS,
    }
    threshold_path = models_dir / THRESHOLD_FILE
    with open(threshold_path, "w", encoding="utf-8") as f:
        json.dump(threshold_info, f, indent=2)

    print(f"Saved model to: {model_path}")
    print(f"Saved threshold metadata to: {threshold_path}")

    return {
        "model": chosen_model,
        "model_path": model_path,
        "threshold_path": threshold_path,
        "best_threshold": best_thresh,
        "best_macro_f05": best_macro_f05,
        "pair_stats": pair_stats,
        "lgb_metrics": lgb_metrics,
        "lr_metrics": lr_metrics,
    }


def main():
    parser = argparse.ArgumentParser(description="Train Entity Matching model and sweep optimal threshold")
    parser.add_argument("--train-dir", default="dataset/train", help="Path to training TSVs directory")
    parser.add_argument("--models-dir", default="models", help="Directory to save matcher.pkl and threshold.json")
    parser.add_argument("--val-size", type=float, default=0.2, help="Validation split ratio")
    args = parser.parse_args()

    train_entity_matching_system(
        data_dir=args.train_dir,
        models_dir=args.models_dir,
        val_size=args.val_size,
    )


if __name__ == "__main__":
    main()