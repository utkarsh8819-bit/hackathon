"""
Top-level pipeline execution and submission packaging script.
"""
import argparse
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from src.train import train_entity_matching_system
from src.inference import run_inference_pipeline


def build_submission_zip(
    output_dir: Path,
    code_dir: Path,
    doc_path: Path,
    zip_path: Path,
) -> None:
    """
    Assembles final submission archive strictly following competition specs:
    <team_name>_submission.zip
    ├── output/
    │   ├── matching_results.tsv
    │   └── candidate_pairs.tsv
    ├── code/
    │   └── business_entity_resolution/
    │       ├── src/
    │       ├── README.md
    │       └── requirements.txt
    └── Documentation_template.md
    """
    print(f"\nPackaging final submission zip: {zip_path}")
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # 1. Output files
        for fname in ["matching_results.tsv", "candidate_pairs.tsv"]:
            fpath = output_dir / fname
            if fpath.exists():
                zf.write(fpath, arcname=f"output/{fname}")
            else:
                print(f"Warning: {fpath} not found for zip packaging")

        # 2. Code files under code/business_entity_resolution/
        code_prefix = "code/business_entity_resolution"
        for fname in ["README.md", "requirements.txt"]:
            fpath = code_dir / fname
            if fpath.exists():
                zf.write(fpath, arcname=f"{code_prefix}/{fname}")

        src_dir = code_dir / "src"
        if src_dir.exists():
            for root, _, files in os.walk(src_dir):
                for f in files:
                    if f.endswith((".py", ".json")):
                        full_f = Path(root) / f
                        rel_f = full_f.relative_to(code_dir)
                        zf.write(full_f, arcname=f"{code_prefix}/{rel_f.as_posix()}")

        # 3. Documentation template
        if doc_path.exists():
            zf.write(doc_path, arcname="Documentation_template.md")

    print(f"Submission zip successfully created at: {zip_path}")


def main():
    parser = argparse.ArgumentParser(description="Run complete Entity Resolution pipeline end-to-end")
    parser.add_argument("--train-dir", default="dataset/train", help="Path to training directory")
    parser.add_argument("--test-dir", default="dataset/test", help="Path to test directory")
    parser.add_argument("--output-dir", default="output", help="Directory for output files")
    parser.add_argument("--models-dir", default="models", help="Directory for saved model and threshold")
    parser.add_argument("--team-name", default="antigravity_team", help="Team name for packaging zip")
    parser.add_argument("--package", action="store_true", help="Generate submission zip after validation")
    parser.add_argument("--skip-train", action="store_true", help="Skip training (use existing saved model)")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    train_dir = Path(args.train_dir)
    test_dir = Path(args.test_dir)
    output_dir = Path(args.output_dir)
    models_dir = Path(args.models_dir)

    print("=" * 60)
    print("AMAZON ML CHALLENGE: BUSINESS ENTITY RESOLUTION PIPELINE")
    print("=" * 60)

    # Step 1: Train model & sweep threshold (if not skipped)
    if not args.skip_train:
        print("\n--- STAGE 1: TRAINING & THRESHOLD OPTIMIZATION ---")
        train_res = train_entity_matching_system(
            data_dir=train_dir,
            models_dir=models_dir,
        )

    # Step 2: Run inference on test data
    print("\n--- STAGE 2: INFERENCE & SUBMISSION GENERATION ---")
    match_path, cand_path = run_inference_pipeline(
        test_dir=test_dir,
        output_dir=output_dir,
        models_dir=models_dir,
    )

    # Step 3: Run submission validator
    print("\n--- STAGE 3: VALIDATING SUBMISSION FORMAT ---")
    validator_path = project_root / "utils" / "validate_submission.py"
    val_cmd = [
        sys.executable,
        str(validator_path),
        "--matching", str(match_path),
        "--candidate", str(cand_path),
        "--test-dir", str(test_dir),
    ]
    val_proc = subprocess.run(val_cmd, capture_output=True, text=True)
    print(val_proc.stdout)
    if val_proc.stderr:
        print(val_proc.stderr)

    if val_proc.returncode != 0:
        print("Validation FAILED! Please resolve the listed issues.")
        sys.exit(1)

    print("Validation PASSED successfully.")

    # Step 4: Package if requested
    if args.package:
        zip_path = project_root / f"{args.team_name}_submission.zip"
        build_submission_zip(
            output_dir=output_dir,
            code_dir=project_root,
            doc_path=project_root / "Documentation_template.md",
            zip_path=zip_path,
        )


if __name__ == "__main__":
    main()
