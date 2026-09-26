# Kaggle Notebook Cells
# Instructions: create a Python notebook on kaggle.com, add the three
# private datasets (amazon-ml-2026-code, amazon-ml-2026-data,
# amazon-ml-2026-interim), paste Cell 1 and Cell 2 below, then
# Save Version -> Run All (background).

## Cell 1 — Setup

    import os, shutil, sys

    !pip install -q rapidfuzz indic-transliteration

    CODE_DIR   = "/kaggle/input/amazon-ml-2026-code"
    DATA_SRC   = "/kaggle/input/amazon-ml-2026-data"
    INTERIM_SRC= "/kaggle/input/amazon-ml-2026-interim"
    WORK       = "/kaggle/working"

    # Locate the code root (src/ may be at top level or nested)
    code_root = None
    for base, dirs, files in os.walk(CODE_DIR):
        if "src" in dirs and os.path.isdir(os.path.join(base, "src", "entity_resolution")):
            code_root = base
            break
    assert code_root, "Could not find src/entity_resolution under CODE_DIR"
    print("code_root =", code_root)

    # Prepare working dataset layout
    DATASET_DIR = f"{WORK}/student_resource/dataset"
    INTERIM_DIR = f"{WORK}/data/interim"
    OUTPUT_DIR  = f"{WORK}/output"
    os.makedirs(f"{DATASET_DIR}/train", exist_ok=True)
    os.makedirs(f"{DATASET_DIR}/test",  exist_ok=True)
    os.makedirs(INTERIM_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Copy TSVs (find them wherever they landed under DATA_SRC)
    for base, dirs, files in os.walk(DATA_SRC):
        for f in files:
            if not f.endswith(".tsv"):
                continue
            src_path = os.path.join(base, f)
            if "train" in f:
                shutil.copy(src_path, f"{DATASET_DIR}/train/{f}")
            elif "test" in f:
                shutil.copy(src_path, f"{DATASET_DIR}/test/{f}")

    # Copy interim artifacts (walker: handles nested zip layout)
    for base, dirs, files in os.walk(INTERIM_SRC):
        for f in files:
            shutil.copy(os.path.join(base, f), f"{INTERIM_DIR}/{f}")

    os.environ["AMAZON_ML_ROOT"] = WORK
    sys.path.insert(0, code_root)

    print("work =", WORK)
    print("dataset_dir =", DATASET_DIR)
    print("interim_dir =", INTERIM_DIR)
    print("train files:", os.listdir(f"{DATASET_DIR}/train"))
    print("test files :", os.listdir(f"{DATASET_DIR}/test"))
    print("interim    :", os.listdir(INTERIM_DIR))

## Cell 2 — Run inference

    import os
    os.chdir("/kaggle/working")
    # Run as a module from /kaggle/working with code on PYTHONPATH
    import sys
    sys.path.insert(0, code_root)          # code_root from Cell 1
    !python -m src.entity_resolution.inference || \
        python {code_root}/src/entity_resolution/inference.py