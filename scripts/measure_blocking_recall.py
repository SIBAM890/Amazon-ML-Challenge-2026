import os
import sys
import gc
import numpy as np
import pandas as pd
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.entity_resolution.config import (
    TRAIN_SOURCE1, TRAIN_SOURCE2, TRAIN_SOURCE3, TRAIN_GROUND_TRUTH
)
from src.entity_resolution.blocking import Blocker


def measure_blocking_recall():
    print("Loading ground truth and S1 (for validation sample)...", flush=True)
    gt_df = pd.read_csv(TRAIN_GROUND_TRUTH, sep='\t', dtype=str)

    # ---- Diagnostics (required before applying the fix) ----
    print("isna count:", gt_df['matched_entity_ids'].isna().sum(), flush=True)
    print("empty-str count:", (gt_df['matched_entity_ids'] == "").sum(), flush=True)
    print("dtypes:", gt_df.dtypes.to_dict(), flush=True)

    # FIX A: singletons are NaN (not ""), so fillna first before filtering.
    has_match = gt_df['matched_entity_ids'].fillna('').astype(str).str.strip() != ''
    filtered_count = has_match.sum()
    print(f"Entities with >= 1 true match: {filtered_count} (expected 2,083,574)", flush=True)
    assert filtered_count == 2_083_574, (
        f"FATAL: filtered count {filtered_count} != expected 2,083,574"
    )

    # Sample 20k S1 entities that have >= 1 true match — recall metric
    # is only meaningful on entities that have a true candidate to recover.
    gt_val = gt_df[has_match].sample(n=20_000, random_state=42)
    del gt_df
    gc.collect()

    val_s1_ids = set(gt_val['source1_entity_id'])
    print(f"Validation sample: {len(gt_val)} S1 entities with >= 1 true match", flush=True)

    # FIX B: y_true built from sample only; denominator is sample-local.
    y_true = {
        row.source1_entity_id: set(row.matched_entity_ids.split(','))
        for row in gt_val.itertuples(index=False)
    }
    total_true_pairs_in_sample = sum(len(v) for v in y_true.values())
    print(f"True matched pairs in 20k sample (denominator): {total_true_pairs_in_sample}", flush=True)

    del gt_val
    gc.collect()

    # Build Blocker using file paths (never loads full S2+S3 into RAM at once)
    print("Building blocker (two-pass chunked index over ~10.3M S2+S3 rows)...", flush=True)
    start_time = time.time()
    blocker = Blocker(TRAIN_SOURCE2, TRAIN_SOURCE3, rare_freq_threshold=500)
    blocker_time = time.time() - start_time
    print(f"Blocker built in {blocker_time:.1f}s", flush=True)

    # Load only the 20k S1 rows we need
    df_s1 = pd.read_csv(TRAIN_SOURCE1, sep='\t', dtype=str)
    df_s1_val = df_s1[df_s1['entity_id'].isin(val_s1_ids)].copy()
    del df_s1
    gc.collect()
    print(f"S1 rows loaded for sample: {len(df_s1_val)}", flush=True)

    # Measure recall — itertuples over 20k rows is fine
    recovered = 0
    total_candidates = 0
    cand_counts = []

    print("Generating candidates...", flush=True)
    start_time = time.time()
    for i, row in enumerate(df_s1_val.itertuples(index=False), 1):
        if i % 5000 == 0:
            elapsed_so_far = time.time() - start_time
            print(f"  processed {i}/20000 S1 entities; "
                  f"candidates so far: {total_candidates}; "
                  f"elapsed: {elapsed_so_far:.1f}s", flush=True)
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
    print(f"Candidate generation: {elapsed:.1f}s", flush=True)

    recall = recovered / total_true_pairs_in_sample if total_true_pairs_in_sample > 0 else 1.0
    c = np.array(cand_counts)

    print("\n========== BLOCKING RECALL RESULTS ==========", flush=True)
    print(f"1. Sampled S1 count:              {len(df_s1_val)}", flush=True)
    print(f"2. Total true matched pairs:      {total_true_pairs_in_sample}", flush=True)
    print(f"3. True pairs found (recovered):  {recovered}", flush=True)
    print(f"4. Blocking recall:               {recall:.6f}", flush=True)
    print(f"5. Mean candidates per S1:        {c.mean():.2f}", flush=True)
    print(f"6. Median candidates per S1:      {np.median(c):.2f}", flush=True)
    print(f"7. P95 candidates per S1:         {np.percentile(c, 95):.2f}", flush=True)
    print(f"8. Maximum candidates per S1:     {c.max()}", flush=True)
    print(f"   Total candidate pairs:         {total_candidates}", flush=True)
    print(f"   Blocker build time:            {blocker_time:.1f}s", flush=True)
    print(f"   Candidate generation time:     {elapsed:.1f}s", flush=True)
    print("=============================================", flush=True)

    if recall < 0.90:
        print("\n*** WARNING: Blocking recall < 0.90! ***", flush=True)
        print("*** DO NOT proceed to model. Diagnose blocking failure first. ***", flush=True)
    elif recall < 0.98:
        print(f"\nBlocking recall {recall:.4f} is >= 0.90 but below 0.98 target.", flush=True)
        print("Consider adding character n-gram retrieval as an additional pass.", flush=True)
    else:
        print(f"\nBlocking recall {recall:.4f} meets the >= 0.98 engineering target.", flush=True)


if __name__ == "__main__":
    measure_blocking_recall()
