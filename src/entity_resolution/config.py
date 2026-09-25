import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Paths to student_resource
STUDENT_RESOURCE_DIR = os.path.join(PROJECT_ROOT, "student_resource")
DATASET_DIR = os.path.join(STUDENT_RESOURCE_DIR, "dataset")

# Train paths
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
TRAIN_SOURCE1 = os.path.join(TRAIN_DIR, "train_source1.tsv")
TRAIN_SOURCE2 = os.path.join(TRAIN_DIR, "train_source2.tsv")
TRAIN_SOURCE3 = os.path.join(TRAIN_DIR, "train_source3.tsv")
TRAIN_GROUND_TRUTH = os.path.join(TRAIN_DIR, "train_ground_truth.tsv")

# Test paths
TEST_DIR = os.path.join(DATASET_DIR, "test")
TEST_SOURCE1 = os.path.join(TEST_DIR, "test_source1.tsv")
TEST_SOURCE2 = os.path.join(TEST_DIR, "test_source2.tsv")
TEST_SOURCE3 = os.path.join(TEST_DIR, "test_source3.tsv")

# Utils
VALIDATE_SUBMISSION_SCRIPT = os.path.join(STUDENT_RESOURCE_DIR, "utils", "validate_submission.py")
