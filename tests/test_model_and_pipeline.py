import subprocess
import sys
import tempfile
from pathlib import Path
import pandas as pd
import pytest

from src.train import train_entity_matching_system
from src.inference import run_inference_pipeline
from src.io import write_tsv, read_tsv
from tests.fixtures_data import generate_sample_train_data, generate_sample_test_data


def test_full_pipeline_train_inference_and_validation():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        train_dir = tmp_path / "dataset" / "train"
        test_dir = tmp_path / "dataset" / "test"
        models_dir = tmp_path / "models"
        output_dir = tmp_path / "output"

        train_dir.mkdir(parents=True)
        test_dir.mkdir(parents=True)

        s1_tr, s2_tr, s3_tr, gt_tr = generate_sample_train_data()
        write_tsv(s1_tr, train_dir / "train_source1.tsv")
        write_tsv(s2_tr, train_dir / "train_source2.tsv")
        write_tsv(s3_tr, train_dir / "train_source3.tsv")
        write_tsv(gt_tr, train_dir / "train_ground_truth.tsv")

        s1_te, s2_te, s3_te = generate_sample_test_data()
        write_tsv(s1_te, test_dir / "test_source1.tsv")
        write_tsv(s2_te, test_dir / "test_source2.tsv")
        write_tsv(s3_te, test_dir / "test_source3.tsv")

        # 1. Train model
        train_res = train_entity_matching_system(
            data_dir=train_dir,
            models_dir=models_dir,
            val_size=0.4,
            random_state=42,
        )
        assert (models_dir / "matcher.pkl").exists()
        assert (models_dir / "threshold.json").exists()
        assert train_res["best_macro_f05"] >= 0.0

        # 2. Run inference
        match_path, cand_path = run_inference_pipeline(
            test_dir=test_dir,
            output_dir=output_dir,
            models_dir=models_dir,
        )
        assert match_path.exists()
        assert cand_path.exists()

        # Check structural properties
        match_df = read_tsv(match_path)
        cand_df = read_tsv(cand_path)
        assert len(match_df) == len(s1_te)
        assert len(cand_df) == len(s1_te)
        assert "S1-9001" in match_df["source1_entity_id"].values  # French entity included!

        # 3. Run submission validator script
        validator_script = Path(__file__).resolve().parent.parent / "utils" / "validate_submission.py"
        cmd = [
            sys.executable,
            str(validator_script),
            "--matching", str(match_path),
            "--candidate", str(cand_path),
            "--test-dir", str(test_dir),
            "--check-ids",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        print("Validator stdout:\n", proc.stdout)
        print("Validator stderr:\n", proc.stderr)
        assert proc.returncode == 0
        assert "PASS" in proc.stdout
