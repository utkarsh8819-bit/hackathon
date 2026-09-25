import tempfile
from pathlib import Path
from src.blocking import BlockingEngine
from src.output import export_candidate_pairs
from src.io import read_tsv
from tests.fixtures_data import generate_sample_train_data


def test_blocking_and_candidate_generation():
    s1_df, s2_df, s3_df, gt_df = generate_sample_train_data()
    engine = BlockingEngine(max_candidates_per_query=50)
    engine.fit(s2_df, s3_df)

    internal_df, comp_df = engine.generate_candidates(s1_df)

    assert len(comp_df) == len(s1_df)
    assert "source1_entity_id" in comp_df.columns
    assert "candidate_entity_ids" in comp_df.columns

    # Verify that multi-match S1-1001 retrieved both S2-2001 and S3-3001
    s1_1001_cands = comp_df[comp_df["source1_entity_id"] == "S1-1001"]["candidate_entity_ids"].iloc[0].split(",")
    assert "S2-2001" in s1_1001_cands
    assert "S3-3001" in s1_1001_cands

    # Verify Indian address with landmark/PIN match S1-1002 retrieved S2-2002 and S3-3002
    s1_1002_cands = comp_df[comp_df["source1_entity_id"] == "S1-1002"]["candidate_entity_ids"].iloc[0].split(",")
    assert "S2-2002" in s1_1002_cands
    assert "S3-3002" in s1_1002_cands

    # Check blocking recall on ground truth
    gt_pairs = set()
    for _, row in gt_df.iterrows():
        s1 = row["source1_entity_id"]
        matches = [m.strip() for m in str(row["matched_entity_ids"]).split(",") if m.strip()]
        for m in matches:
            gt_pairs.add((s1, m))

    candidate_pairs = set(zip(internal_df["source1_entity_id"], internal_df["candidate_entity_id"]))
    survived = gt_pairs.intersection(candidate_pairs)
    recall = len(survived) / len(gt_pairs) if gt_pairs else 1.0

    print(f"Blocking recall on synthetic fixture: {recall:.4f} ({len(survived)}/{len(gt_pairs)})")
    assert recall >= 0.95


def test_export_candidate_pairs_format():
    s1_df, s2_df, s3_df, _ = generate_sample_train_data()
    engine = BlockingEngine()
    engine.fit(s2_df, s3_df)
    internal_df, _ = engine.generate_candidates(s1_df)

    with tempfile.TemporaryDirectory() as tmpdir:
        comp_path, int_path = export_candidate_pairs(
            internal_df,
            s1_entity_ids=s1_df["entity_id"].tolist(),
            output_dir=tmpdir,
            save_internal_4col=True,
        )

        assert comp_path.exists()
        assert int_path.exists()

        comp_loaded = read_tsv(comp_path)
        assert list(comp_loaded.columns) == ["source1_entity_id", "candidate_entity_ids"]
        assert len(comp_loaded) == len(s1_df)

        int_loaded = read_tsv(int_path)
        assert list(int_loaded.columns) == [
            "source1_entity_id",
            "candidate_entity_id",
            "candidate_source",
            "blocking_reasons",
        ]
