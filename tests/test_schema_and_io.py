import tempfile
from pathlib import Path
import pytest
import pandas as pd

from src.io import read_tsv, write_tsv, load_source_file, load_ground_truth_file
from src.validation import validate_source_dataframe, parse_ground_truth, ValidationError
from tests.fixtures_data import generate_sample_train_data


def test_tsv_roundtrip_with_commas():
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = Path(tmpdir) / "test.tsv"
        data = pd.DataFrame([
            {"entity_id": "S1-001", "business_name": "Smith, Jones & Co", "business_address": "123 Main St, Apt 4, Boston, MA", "country": "US"}
        ])
        write_tsv(data, file_path)
        loaded = read_tsv(file_path)
        assert len(loaded) == 1
        assert loaded.iloc[0]["business_name"] == "Smith, Jones & Co"
        assert loaded.iloc[0]["business_address"] == "123 Main St, Apt 4, Boston, MA"


def test_validate_source_invalid_prefix():
    bad_df = pd.DataFrame([
        {"entity_id": "WRONG-001", "business_name": "Test", "business_address": "Addr", "country": "US"}
    ])
    with pytest.raises(ValidationError):
        validate_source_dataframe(bad_df, expected_prefix="S1-", dataset_name="bad_s1")


def test_validate_source_duplicate_ids():
    dup_df = pd.DataFrame([
        {"entity_id": "S1-001", "business_name": "Test1", "business_address": "Addr1", "country": "US"},
        {"entity_id": "S1-001", "business_name": "Test2", "business_address": "Addr2", "country": "US"},
    ])
    with pytest.raises(ValidationError):
        validate_source_dataframe(dup_df, expected_prefix="S1-", dataset_name="dup_s1")


def test_parse_ground_truth_singletons_and_multimatch():
    gt_df = pd.DataFrame([
        {"source1_entity_id": "S1-001", "matched_entity_ids": "S2-101,S3-201"},
        {"source1_entity_id": "S1-002", "matched_entity_ids": ""}, # Singleton
        {"source1_entity_id": "S1-003", "matched_entity_ids": "S2-102"},
    ])
    gt_map, stats = parse_ground_truth(gt_df)
    assert len(gt_map) == 3
    assert gt_map["S1-001"] == ["S2-101", "S3-201"]
    assert gt_map["S1-002"] == []
    assert gt_map["S1-003"] == ["S2-102"]
    assert stats["singletons_count"] == 1
    assert stats["multi_match_count"] == 1


def test_parse_ground_truth_rejects_self_match():
    bad_gt = pd.DataFrame([
        {"source1_entity_id": "S1-001", "matched_entity_ids": "S1-001"}
    ])
    with pytest.raises(ValidationError):
        parse_ground_truth(bad_gt)
