"""
Input/Output utilities for TSV files and canonical dataset representation.
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd

from src.config import (
    SOURCE_COLUMNS,
    GROUND_TRUTH_COLUMNS,
    TRAIN_SOURCE1,
    TRAIN_SOURCE2,
    TRAIN_SOURCE3,
    TRAIN_GROUND_TRUTH,
    TEST_SOURCE1,
    TEST_SOURCE2,
    TEST_SOURCE3,
)
from src.validation import validate_source_dataframe, parse_ground_truth


def read_tsv(filepath: Union[str, Path]) -> pd.DataFrame:
    """
    Safely reads a TSV file with tab separator, preserving literal commas,
    avoiding silent column collapsing, and keeping empty strings.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    df = pd.read_csv(
        filepath,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        encoding="utf-8",
    )
    return df


def write_tsv(df: pd.DataFrame, filepath: Union[str, Path], columns: Optional[List[str]] = None) -> None:
    """
    Safely writes a DataFrame to a TSV file with tab separator and UTF-8 encoding.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    if columns is not None:
        df = df[columns]
    df.to_csv(filepath, sep="\t", index=False, encoding="utf-8")


def load_source_file(
    filepath: Union[str, Path],
    source_prefix: str,
    strict: bool = True,
) -> pd.DataFrame:
    """
    Loads one source file (S1, S2, or S3), validates schema, and attaches canonical
    metadata columns (source: 'S1'/'S2'/'S3', record_role: 'query'/'candidate').
    Preserves raw text fields unchanged.
    """
    df = read_tsv(filepath)
    validate_source_dataframe(df, expected_prefix=source_prefix, dataset_name=Path(filepath).name, strict=strict)
    
    # Add canonical representation columns
    df["source"] = source_prefix.replace("-", "")
    df["record_role"] = "query" if source_prefix.startswith("S1") else "candidate"
    
    # Ensure all required columns are string and non-null
    for col in SOURCE_COLUMNS:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str).str.strip()
            
    return df


def load_ground_truth_file(
    filepath: Union[str, Path],
    valid_s1_ids: Optional[set] = None,
    valid_s2_s3_ids: Optional[set] = None,
    strict: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, List[str]], Dict[str, any]]:
    """
    Loads ground truth TSV, validates, and returns (df, gt_map, stats).
    """
    df = read_tsv(filepath)
    gt_map, stats = parse_ground_truth(df, valid_s1_ids=valid_s1_ids, valid_s2_s3_ids=valid_s2_s3_ids, strict=strict)
    return df, gt_map, stats


def load_split_datasets(
    data_dir: Union[str, Path],
    is_train: bool = True,
    strict: bool = True,
) -> Dict[str, any]:
    """
    Loads all source files (and ground truth if is_train=True) for a split.
    """
    data_dir = Path(data_dir)
    if is_train:
        s1 = load_source_file(data_dir / TRAIN_SOURCE1, source_prefix="S1-", strict=strict)
        s2 = load_source_file(data_dir / TRAIN_SOURCE2, source_prefix="S2-", strict=strict)
        s3 = load_source_file(data_dir / TRAIN_SOURCE3, source_prefix="S3-", strict=strict)
        
        valid_s1 = set(s1["entity_id"])
        valid_s2_s3 = set(s2["entity_id"]) | set(s3["entity_id"])
        gt_df, gt_map, gt_stats = load_ground_truth_file(
            data_dir / TRAIN_GROUND_TRUTH,
            valid_s1_ids=valid_s1,
            valid_s2_s3_ids=valid_s2_s3,
            strict=strict,
        )
        return {
            "source1": s1,
            "source2": s2,
            "source3": s3,
            "ground_truth_df": gt_df,
            "ground_truth_map": gt_map,
            "ground_truth_stats": gt_stats,
        }
    else:
        s1 = load_source_file(data_dir / TEST_SOURCE1, source_prefix="S1-", strict=strict)
        s2 = load_source_file(data_dir / TEST_SOURCE2, source_prefix="S2-", strict=strict)
        s3 = load_source_file(data_dir / TEST_SOURCE3, source_prefix="S3-", strict=strict)
        return {
            "source1": s1,
            "source2": s2,
            "source3": s3,
        }
