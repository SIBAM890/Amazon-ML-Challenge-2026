"""Test retrieval K sweep on the 20k validation sample (foreground).

Loads data/interim/candidates_20k_sample_v2.parquet (blocking pools) +
data/interim/gt_20k_sample.parquet (ground truth), builds an
S2/S3 lookup chunked from train TSVs, then runs retrieve() for
K in {25, 50, 100} and reports recall + mean/median/p95 per S1.
"""
import gc
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.entity_resolution.config import (
    TRAIN_SOURCE1,
    TRAIN_SOURCE2,
    TRAIN_SOURCE3,
)
from src.entity_resolution.retrieval import retrieve


def _split_ids(s: object) -> list:
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return []
    s = s if isinstance(s, str) else str(s)
    if not s or s == "nan":
        return []
    return [c for c in s.split(",") if c]


def main() -> None:
    t0 = time.time()
    print("loading parquets...", flush=True)
    gt = pd.read_parquet("data/interim/gt_20k_sample.parquet")
    v2 = pd.read_parquet("data/interim/candidates_20k_sample_v2.parquet")
    print(f"gt={len(gt)} v2={len(v2)} t={time.time()-t0:.1f}s", flush=True)

    y_true = {
        r.source1_entity_id: set(_split_ids(r.matched_entity_ids))
        for r in gt.itertuples(index=False)
    }
    pools = {
        r.source1_entity_id: _split_ids(r.candidate_entity_ids)
        for r in v2.itertuples(index=False)
    }
    del gt, v2
    gc.collect()
    denom = sum(len(v) for v in y_true.values())
    print(f"true pairs (denominator)={denom}", flush=True)

    s1_ids = set(y_true.keys())
    cand_ids: set = set()
    for lst in pools.values():
        cand_ids.update(lst)
    need_s2 = {c for c in cand_ids if c.startswith("S2-")}
    need_s3 = {c for c in cand_ids if c.startswith("S3-")}
    print(f"s1={len(s1_ids)} uniq_cand={len(cand_ids)} "
          f"s2={len(need_s2)} s3={len(need_s3)}", flush=True)

    print("loading S1 records...", flush=True)
    s1rec: dict = {}
    for ch in pd.read_csv(TRAIN_SOURCE1, sep="\t", dtype=str,
                          usecols=["entity_id", "business_name",
                                   "business_address", "country"],
                          chunksize=500_000):
        sub = ch[ch["entity_id"].isin(s1_ids)]
        for r in sub.itertuples(index=False):
            s1rec[r.entity_id] = {
                "entity_id": r.entity_id,
                "business_name": r.business_name,
                "business_address": r.business_address,
                "country": r.country,
            }
        del ch
        gc.collect()
        if len(s1rec) >= len(s1_ids):
            break
    print(f"s1rec={len(s1rec)} t={time.time()-t0:.1f}s", flush=True)

    print("loading S2/S3 lookup...", flush=True)
    lookup: dict = {}
    for path, need in ((TRAIN_SOURCE2, need_s2), (TRAIN_SOURCE3, need_s3)):
        for ch in pd.read_csv(path, sep="\t", dtype=str,
                              usecols=["entity_id", "business_name",
                                       "business_address", "country"],
                              chunksize=500_000):
            sub = ch[ch["entity_id"].isin(need)]
            for r in sub.itertuples(index=False):
                lookup[r.entity_id] = {
                    "business_name": r.business_name,
                    "business_address": r.business_address,
                    "country": r.country,
                }
            del ch
            gc.collect()
            if len(lookup) >= len(need_s2) + len(need_s3):
                break
        print(f"  lookup={len(lookup)} t={time.time()-t0:.1f}s", flush=True)
    print(f"lookup complete={len(lookup)} t={time.time()-t0:.1f}s", flush=True)

    # -- blocking baseline (pre-retrieval) -------------------------------
    rec0 = 0
    sizes0 = []
    for sid, true_m in y_true.items():
        pool = set(pools.get(sid, []))
        sizes0.append(len(pool))
        rec0 += len(pool & true_m)
    a0 = np.array(sizes0)
    print(f"[baseline blocking] recall={rec0/denom:.4f} "
          f"mean={a0.mean():.1f} median={np.median(a0):.1f} "
          f"p95={np.percentile(a0, 95):.1f} total={a0.sum()} "
          f"t={time.time()-t0:.1f}s", flush=True)

    # -- K sweep (one retrieve() pass per S1 per K) -----------------------
    for K in (25, 50, 100):
        tk = time.time()
        rec = 0
        sizes = []
        for i, (sid, true_m) in enumerate(y_true.items(), 1):
            if i % 5000 == 0:
                print(f"  [K={K}] {i}/20000 elapsed={time.time()-tk:.1f}s",
                      flush=True)
            srow = s1rec.get(sid)
            if srow is None:
                sizes.append(0)
                continue
            final, _meta = retrieve(srow, pools.get(sid, []), lookup, K=K)
            sizes.append(len(final))
            rec += len(final & true_m)
        a = np.array(sizes)
        print(f"[K={K}] recall={rec/denom:.4f} "
              f"mean={a.mean():.1f} median={np.median(a):.1f} "
              f"p95={np.percentile(a, 95):.1f} total={a.sum()} "
              f"sweep_t={time.time()-tk:.1f}s all_t={time.time()-t0:.1f}s",
              flush=True)


if __name__ == "__main__":
    main()
