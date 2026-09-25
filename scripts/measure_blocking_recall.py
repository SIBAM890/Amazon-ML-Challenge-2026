import os
import sys
import numpy as np
import pandas as pd
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.entity_resolution.config import (
    TRAIN_SOURCE1, TRAIN_SOURCE2, TRAIN_SOURCE3, TRAIN_GROUND_TRUTH
)
from src.entity_resolution.blocking import Blocker


def measure_blocking_recall():
    print("Loading ground truth and S1 (for validation sample)...")
    gt_df = pd.read_csv(TRAIN_GROUND_TRUTH, sep='\t', dtype=str)
    # FIX A: singletons are NaN (not ""), so fillna first before filtering.
    # Diagnostic confirmed: isna count=123247, empty-str count=0.
    has_match = gt_df['matched_entity_ids'].fillna('').astype(str).str.strip() != ''
    print(f"Entities with >= 1 true match: {has_match.sum()} (expected 2,083,574)")

    # Sample 20k S1 entities that have >= 1 true match — recall metric
    # is only meaningful on entities that have a true candidate to recover.
    gt_val = gt_df[has_match].sample(n=20_000, random_state=42)
    del gt_df

    val_s1_ids = set(gt_val['source1_entity_id'])
    print(f"Validation sample: {len(gt_val)} S1 entities with >= 1 true match")
    # FIX B: y_true built from sample only; denominator is sample-local.
    y_true = {
        row.source1_entity_id: set(row.matched_entity_ids.split(','))
        for row in gt_val.itertuples(index=False)
    }
    total_true_pairs_in_sample = sum(len(v) for v in y_true.values())
    print(f"True matched pairs in 20k sample (denominator): {total_true_pairs_in_sample}")

    # Build Blocker using file paths (never loads full S2+S3 into RAM at once)
    print("Building blocker (this takes ~20 min on first run)...")
    start_time = time.time()
    blocker = Blocker(TRAIN_SOURCE2, TRAIN_SOURCE3, rare_freq_threshold=500)
    print(f"Blocker built in {time.time() - start_time:.1f}s")

    # Load only the 20k S1 rows we need
    df_s1 = pd.read_csv(TRAIN_SOURCE1, sep='\t', dtype=str)
    df_s1_val = df_s1[df_s1['entity_id'].isin(val_s1_ids)].copy()
    del df_s1
    print(f"S1 rows loaded for sample: {len(df_s1_val)}")

    # Measure recall — itertuples over 20k rows is fine
    recovered = 0
    total_candidates = 0
    cand_counts = []

    print("Generating candidates...")
    start_time = time.time()
    for i, row in enumerate(df_s1_val.itertuples(index=False), 1):
        if i % 5000 == 0:
            print(f"processed {i} of 20000 S1 entities; running candidate total: {total_candidates}")
        cands = blocker.get_candidates(
            pd.Series({
                'entity_id': row.entity_id,
                'business_name': row.business_name,
                'business_address': row.business_address,
                'country': row.country,
            })
        )
        n = len(cands)
        total_candidates += n
        cand_counts.append(n)
        true_m = y_true.get(row.entity_id, set())
        recovered += len(cands.intersection(true_m))

    elapsed = time.time() - start_time
    print(f"Candidate generation: {elapsed:.1f}s")

    recall = recovered / total_true_pairs_in_sample if total_true_pairs_in_sample > 0 else 1.0
    c = np.array(cand_counts)

    print("\n--- Blocking Results ---")
    print(f"Sample size: {len(df_s1_val)} S1 entities")
    print(f"Total candidate pairs: {total_candidates}")
    print(f"Recall: {recall:.4f}")
    print(f"Mean candidates per S1:   {c.mean():.2f}")
    print(f"Median candidates per S1: {np.median(c):.2f}")
    print(f"P95 candidates per S1:    {np.percentile(c, 95):.2f}")


if __name__ == "__main__":
    measure_blocking_recall()
