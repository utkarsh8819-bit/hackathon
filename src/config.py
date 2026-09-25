"""
Central configuration for Entity Resolution system.
"""
from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent if "__file__" in globals() else Path(".").resolve()
DATASET_DIR = PROJECT_ROOT / "dataset"
TRAIN_DIR = DATASET_DIR / "train"
TEST_DIR = DATASET_DIR / "test"
OUTPUT_DIR = PROJECT_ROOT / "output"
MODELS_DIR = PROJECT_ROOT / "models"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
UTILS_DIR = PROJECT_ROOT / "utils"

# Standard Filenames
TRAIN_SOURCE1 = "train_source1.tsv"
TRAIN_SOURCE2 = "train_source2.tsv"
TRAIN_SOURCE3 = "train_source3.tsv"
TRAIN_GROUND_TRUTH = "train_ground_truth.tsv"

TEST_SOURCE1 = "test_source1.tsv"
TEST_SOURCE2 = "test_source2.tsv"
TEST_SOURCE3 = "test_source3.tsv"

MATCHING_RESULTS_FILE = "matching_results.tsv"
CANDIDATE_PAIRS_FILE = "candidate_pairs.tsv"
INTERNAL_CANDIDATE_FILE = "candidate_pairs_internal.tsv"

MODEL_FILE = "matcher.pkl"
THRESHOLD_FILE = "threshold.json"

# Schemas
SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]
GROUND_TRUTH_COLUMNS = ["source1_entity_id", "matched_entity_ids"]
MATCHING_HEADER = ["source1_entity_id", "matched_entity_ids"]
CANDIDATE_HEADER = ["source1_entity_id", "candidate_entity_ids"]
INTERNAL_CANDIDATE_HEADER = [
    "source1_entity_id",
    "candidate_entity_id",
    "candidate_source",
    "blocking_reasons",
]

# Pipeline Settings
RANDOM_STATE = 42
F_BETA = 0.5
DEFAULT_MATCH_THRESHOLD = 0.50
MAX_CANDIDATES_PER_S1 = 100
HARD_NEGATIVES_PER_POS = 8
