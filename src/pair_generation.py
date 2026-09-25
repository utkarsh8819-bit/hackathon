"""
Training pair construction: positives, hard negatives, and grouped train/val split.
"""
from typing import Dict, List, Set, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from src.config import RANDOM_STATE, HARD_NEGATIVES_PER_POS


def build_labeled_training_pairs(
    internal_candidates_df: pd.DataFrame,
    gt_map: Dict[str, List[str]],
    hard_negatives_per_pos: int = HARD_NEGATIVES_PER_POS,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, Dict[str, any]]:
    """
    Constructs labeled pairs from internal candidate table and ground truth:
    - Positive: (s1_id, cand_id) where cand_id in gt_map[s1_id]
    - Hard Negative: (s1_id, cand_id) retrieved by blocking but not in gt_map[s1_id]
    - Computes blocking recall ceiling: survived_positives / total_gt_positives.
    """
    total_gt_positives = sum(len(matches) for matches in gt_map.values())
    gt_pos_set = {(s1, mid) for s1, matches in gt_map.items() for mid in matches}

    positive_rows = []
    negative_rows = []

    # Map candidate rows by s1_id for hard negative sampling
    s1_to_candidates = internal_candidates_df.groupby("source1_entity_id")

    rng = np.random.default_rng(random_state)
    survived_positives = 0

    for s1_id, group in s1_to_candidates:
        gt_matches = set(gt_map.get(s1_id, []))
        
        pos_in_group = []
        neg_in_group = []

        for _, row in group.iterrows():
            cand_id = row["candidate_entity_id"]
            pair = (s1_id, cand_id)
            pair_dict = row.to_dict()

            if cand_id in gt_matches:
                pair_dict["label"] = 1
                pos_in_group.append(pair_dict)
                survived_positives += 1
            else:
                pair_dict["label"] = 0
                neg_in_group.append(pair_dict)

        positive_rows.extend(pos_in_group)

        # Hard negative subsampling if negatives outnumber target ratio
        num_pos = len(pos_in_group)
        target_negs = max(hard_negatives_per_pos, num_pos * hard_negatives_per_pos)
        
        if len(neg_in_group) > target_negs:
            # Sort by number of blocking reasons (hardest negatives first)
            neg_in_group.sort(
                key=lambda x: len(str(x.get("blocking_reasons", "")).split(",")),
                reverse=True,
            )
            selected_negs = neg_in_group[:target_negs]
        else:
            selected_negs = neg_in_group

        negative_rows.extend(selected_negs)

    all_pairs = positive_rows + negative_rows
    labeled_df = pd.DataFrame(all_pairs)

    recall_ceiling = (
        float(survived_positives / total_gt_positives) if total_gt_positives > 0 else 1.0
    )

    diagnostics = {
        "total_gt_positives": total_gt_positives,
        "survived_positives": survived_positives,
        "missed_positives": total_gt_positives - survived_positives,
        "blocking_recall_ceiling": recall_ceiling,
        "total_labeled_pairs": len(labeled_df),
        "positive_count": len(positive_rows),
        "negative_count": len(negative_rows),
    }

    return labeled_df, diagnostics


def split_grouped_train_val(
    labeled_df: pd.DataFrame,
    val_size: float = 0.2,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits labeled pairs into Train and Validation sets grouped by source1_entity_id.
    Zero data leakage: all candidate pairs for an S1 entity belong exclusively to either split.
    """
    # Ensure train_df contains both classes if the dataset has multiple classes
    for seed in range(random_state, random_state + 15):
        gss = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
        groups = labeled_df["source1_entity_id"]
        train_idx, val_idx = next(gss.split(labeled_df, groups=groups))
        train_df = labeled_df.iloc[train_idx].copy().reset_index(drop=True)
        val_df = labeled_df.iloc[val_idx].copy().reset_index(drop=True)
        if train_df["label"].nunique() >= 2 or labeled_df["label"].nunique() < 2:
            break

    # Assert complete disjointness
    train_s1 = set(train_df["source1_entity_id"])
    val_s1 = set(val_df["source1_entity_id"])
    overlap = train_s1.intersection(val_s1)
    if overlap:
        raise ValueError(f"Data leakage detected! S1 IDs in both train and val: {overlap}")

    return train_df, val_df
