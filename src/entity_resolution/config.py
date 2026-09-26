"""Project paths. Reads AMAZON_ML_ROOT env var if set (for Kaggle);
else falls back to the repo root computed from this file's location."""

import os
from pathlib import Path

ROOT = Path(os.environ.get(
    "AMAZON_ML_ROOT",
    Path(__file__).resolve().parents[2],
))

DATA_DIR = ROOT / "student_resource" / "dataset"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"

TRAIN_SOURCE1 = str(TRAIN_DIR / "train_source1.tsv")
TRAIN_SOURCE2 = str(TRAIN_DIR / "train_source2.tsv")
TRAIN_SOURCE3 = str(TRAIN_DIR / "train_source3.tsv")
TRAIN_GROUND_TRUTH = str(TRAIN_DIR / "train_ground_truth.tsv")

TEST_SOURCE1 = str(TEST_DIR / "test_source1.tsv")
TEST_SOURCE2 = str(TEST_DIR / "test_source2.tsv")
TEST_SOURCE3 = str(TEST_DIR / "test_source3.tsv")

INTERIM_DIR = ROOT / "data" / "interim"
OUTPUT_DIR = ROOT / "output"