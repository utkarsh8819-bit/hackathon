"""
Evaluation metrics: exact Per-Entity Macro F_0.5 score and pair-level metrics.
"""
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)

from src.config import F_BETA


def compute_entity_f_beta(
    true_ids: Set[str],
    pred_ids: Set[str],
    beta: float = F_BETA,
) -> float:
    """
    Computes per-entity F-beta score (default beta=0.5, precision-weighted 2x).
    Singletons:
      - True empty + Pred empty -> 1.0
      - True empty + Pred non-empty -> 0.0
    Non-singletons:
      - True non-empty + Pred empty -> 0.0
      - True non-empty + Pred non-empty -> (1 + beta^2)*P*R / (beta^2*P + R)
    """
    is_true_singleton = len(true_ids) == 0
    is_pred_singleton = len(pred_ids) == 0

    if is_true_singleton:
        return 1.0 if is_pred_singleton else 0.0

    if is_pred_singleton:
        return 0.0

    # Non-singleton case
    tp = len(true_ids.intersection(pred_ids))
    if tp == 0:
        return 0.0

    precision = float(tp / len(pred_ids))
    recall = float(tp / len(true_ids))

    beta_sq = beta * beta
    numerator = (1.0 + beta_sq) * precision * recall
    denominator = (beta_sq * precision) + recall

    return float(numerator / denominator) if denominator > 0 else 0.0


def compute_macro_f_beta(
    gt_map: Dict[str, List[str]],
    pred_map: Dict[str, List[str]],
    all_s1_ids: List[str],
    beta: float = F_BETA,
) -> Tuple[float, Dict[str, any]]:
    """
    Computes the official competition metric:
    Per-entity F_0.5 macro-averaged across ALL Source 1 entities in the evaluation set.
    """
    scores = []
    singleton_correct = 0
    singleton_total = 0
    non_singleton_total = 0

    for s1_id in all_s1_ids:
        true_set = set(gt_map.get(s1_id, []))
        pred_set = set(pred_map.get(s1_id, []))

        score = compute_entity_f_beta(true_set, pred_set, beta=beta)
        scores.append(score)

        if len(true_set) == 0:
            singleton_total += 1
            if len(pred_set) == 0:
                singleton_correct += 1
        else:
            non_singleton_total += 1

    macro_score = float(np.mean(scores)) if scores else 0.0
    
    breakdown = {
        "macro_f_beta": macro_score,
        "total_entities": len(all_s1_ids),
        "singleton_total": singleton_total,
        "singleton_correct": singleton_correct,
        "singleton_accuracy": (singleton_correct / singleton_total) if singleton_total else 1.0,
        "non_singleton_total": non_singleton_total,
    }

    return macro_score, breakdown


def compute_pair_level_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Computes standard binary classification diagnostics on candidate pairs.
    """
    y_pred = (y_prob >= threshold).astype(int)
    
    metrics = {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": 0.0,
        "pr_auc": 0.0,
    }
    
    if len(np.unique(y_true)) > 1:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        except Exception:
            metrics["roc_auc"] = 0.0

        try:
            metrics["pr_auc"] = float(average_precision_score(y_true, y_prob))
        except Exception:
            metrics["pr_auc"] = 0.0

    return metrics


def sweep_decision_threshold(
    val_pairs_df: pd.DataFrame,
    val_probs: np.ndarray,
    gt_map: Dict[str, List[str]],
    val_s1_ids: List[str],
    thresholds: Optional[np.ndarray] = None,
) -> Tuple[float, float, List[Dict[str, float]]]:
    """
    Sweeps decision threshold on held-out validation set to find the threshold
    that directly maximizes the competition metric: per-entity macro F_0.5.
    """
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 37)  # step ~0.025

    val_df_scored = val_pairs_df.copy()
    val_df_scored["score"] = val_probs

    best_thresh = 0.50
    best_macro = -1.0
    sweep_results = []

    for t in thresholds:
        pred_map: Dict[str, List[str]] = {s1: [] for s1 in val_s1_ids}
        
        # Keep pairs where score >= threshold
        accepted = val_df_scored[val_df_scored["score"] >= t]
        grouped = accepted.groupby("source1_entity_id")["candidate_entity_id"].apply(list)
        
        for s1_id, cands in grouped.items():
            if s1_id in pred_map:
                pred_map[s1_id] = cands

        macro_score, _ = compute_macro_f_beta(gt_map, pred_map, val_s1_ids, beta=F_BETA)
        sweep_results.append({"threshold": float(t), "macro_f05": macro_score})

        if macro_score > best_macro:
            best_macro = macro_score
            best_thresh = float(t)

    return best_thresh, best_macro, sweep_results
