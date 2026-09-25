import pytest
from src.evaluate import compute_entity_f_beta, compute_macro_f_beta


def test_readme_f05_example():
    true_set = {"S2-00047", "S3-00812"}
    pred_set = {"S2-00047", "S2-00193", "S3-00812"}
    # P = 2/3, R = 2/2 = 1.0
    # F_0.5 = (1.25 * (2/3) * 1) / (0.25 * (2/3) + 1) = (5/6) / (7/6) = 5/7 ~= 0.7142857
    score = compute_entity_f_beta(true_set, pred_set, beta=0.5)
    assert pytest.approx(score, rel=1e-3) == 0.7142857


def test_singleton_scoring():
    # Correct singleton prediction
    assert compute_entity_f_beta(set(), set(), beta=0.5) == 1.0

    # False merge on singleton (predicted match when none exists)
    assert compute_entity_f_beta(set(), {"S2-999"}, beta=0.5) == 0.0

    # Missed match (true match exists, predicted none)
    assert compute_entity_f_beta({"S2-111"}, set(), beta=0.5) == 0.0


def test_macro_f_beta_averaging():
    gt_map = {
        "S1-1": ["S2-10"],
        "S1-2": [],         # Singleton
        "S1-3": [],         # Singleton
    }
    pred_map = {
        "S1-1": ["S2-10"],  # Exact match -> 1.0
        "S1-2": [],         # Correct singleton -> 1.0
        "S1-3": ["S3-99"],  # False merge on singleton -> 0.0
    }
    all_s1 = ["S1-1", "S1-2", "S1-3"]
    macro_score, stats = compute_macro_f_beta(gt_map, pred_map, all_s1, beta=0.5)
    # Average of [1.0, 1.0, 0.0] = 2/3
    assert pytest.approx(macro_score, rel=1e-4) == 2.0 / 3.0
    assert stats["singleton_total"] == 2
    assert stats["singleton_correct"] == 1
