import os
import sys
import pandas as pd
import numpy as np
import collections
import random
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.entity_resolution.config import TRAIN_SOURCE1, TRAIN_SOURCE2, TRAIN_SOURCE3, TRAIN_GROUND_TRUTH
from src.entity_resolution.blocking import normalize_text, strip_legal_suffixes, extract_pin_codes

def analyze():
    start_global = time.time()
    
    print("B. ROW-COUNT SANITY CHECK FIRST")
    df_s1 = pd.read_csv(TRAIN_SOURCE1, sep='\t', dtype=str)
    df_s2 = pd.read_csv(TRAIN_SOURCE2, sep='\t', dtype=str)
    df_s3 = pd.read_csv(TRAIN_SOURCE3, sep='\t', dtype=str)
    gt_df = pd.read_csv(TRAIN_GROUND_TRUTH, sep='\t', dtype=str)
    
    print(f"train_source1: {df_s1.shape}")
    print(f"train_source2: {df_s2.shape}")
    print(f"train_source3: {df_s3.shape}")
    print(f"train_ground_truth: {gt_df.shape}")
    print("The '~10 million records' referred to the sum of S2 and S3 records (5M + 5.2M = 10.2M candidates).")
    
    # D. Token frequency distribution
    print("\nD. TOKEN FREQUENCY DISTRIBUTION")
    name_tokens = collections.Counter()
    addr_tokens = collections.Counter()
    
    print("Computing token frequencies on S2+S3...")
    df_candidates = pd.concat([df_s2, df_s3], ignore_index=True)
    
    # We'll just sample 500k rows for token distribution if full takes too long, but let's try full
    for idx, row in df_candidates.iterrows():
        if time.time() - start_global > 110:
            print("TIMEOUT REACHED DURING TOKEN COUNTING!")
            return
            
        if not pd.isna(row['business_name']):
            nm = strip_legal_suffixes(normalize_text(row['business_name']))
            for t in set(nm.split()): name_tokens[t] += 1
            
        if not pd.isna(row['business_address']):
            am = normalize_text(row['business_address'])
            for t in set(am.split()): addr_tokens[t] += 1
            
    for label, counter in [("NAMES", name_tokens), ("ADDRESSES", addr_tokens)]:
        total_distinct = len(counter)
        c500 = sum(1 for v in counter.values() if v >= 500)
        c5000 = sum(1 for v in counter.values() if v >= 5000)
        print(f"\n[{label}]")
        print(f"Distinct tokens across S2+S3: {total_distinct}")
        print(f"Tokens >= 500 records: {c500}")
        print(f"Tokens >= 5000 records: {c5000}")
        print(f"Top 20 tokens: {counter.most_common(20)}")
        
    print("\nBuilding indices for Threshold Sweep...")
    # Build a unified index mapping token -> list of (eid, country)
    # To sweep efficiently, we only need to index tokens that are <= 5000 (max threshold).
    name_token_idx = collections.defaultdict(list)
    exact_name_idx = collections.defaultdict(list)
    pin_idx = collections.defaultdict(list)
    
    # Filter tokens once
    valid_name_tokens = {t: c for t, c in name_tokens.items() if c <= 5000}
    
    for _, row in df_candidates.iterrows():
        if time.time() - start_global > 110:
            print("TIMEOUT REACHED DURING INDEX BUILDING!")
            return
            
        eid, ctry = row['entity_id'], row['country']
        
        # Name tokens
        if not pd.isna(row['business_name']):
            nm = strip_legal_suffixes(normalize_text(row['business_name']))
            if nm: exact_name_idx[(ctry, nm)].append(eid)
            for t in set(nm.split()):
                if t in valid_name_tokens:
                    name_token_idx[(ctry, t)].append(eid)
                    
        # Pins
        pins = extract_pin_codes(row['business_address'])
        for p in pins: pin_idx[(ctry, p)].append(eid)
        
    # C. THRESHOLD SWEEP
    print("\nC. THRESHOLD SWEEP")
    val_size = 5000
    gt_val = gt_df.iloc[-val_size:].copy()
    gt_val['matched_entity_ids'] = gt_val['matched_entity_ids'].fillna("")
    val_s1_ids = set(gt_val['source1_entity_id'])
    df_s1_val = df_s1[df_s1['entity_id'].isin(val_s1_ids)].copy()
    
    y_true = {}
    total_true = 0
    for _, row in gt_val.iterrows():
        s1 = row['source1_entity_id']
        matches_str = row['matched_entity_ids']
        m = set(matches_str.split(',')) if matches_str.strip() else set()
        y_true[s1] = m
        total_true += len(m)
        
    thresholds = [100, 200, 500, 1000, 2000, 5000]
    
    # We will accumulate results across thresholds
    results = {t: {'recovered': 0, 'total_cands': 0, 'cand_counts': [], 'misses': []} for t in thresholds}
    
    print(f"Sweeping {len(df_s1_val)} S1 entities...")
    last_print = time.time()
    
    for i, (_, row) in enumerate(df_s1_val.iterrows()):
        now = time.time()
        if now - start_global > 110:
            print(f"TIMEOUT REACHED AT S1 ENTITY {i}!")
            break
            
        if now - last_print > 5.0:
            print(f"processed {i} of {len(df_s1_val)} S1 entities, candidates so far (at t=500): {results[500]['total_cands']}")
            last_print = now
            
        s1_id = row['entity_id']
        ctry = row['country']
        true_m = y_true.get(s1_id, set())
        
        base_cands = set()
        nm = strip_legal_suffixes(normalize_text(row['business_name']))
        if nm: base_cands.update(exact_name_idx.get((ctry, nm), []))
        
        pins = extract_pin_codes(row['business_address'])
        for p in pins: base_cands.update(pin_idx.get((ctry, p), []))
        
        # Token queries
        s1_tokens = set(nm.split())
        
        for thresh in thresholds:
            t_cands = set(base_cands)
            for t in s1_tokens:
                if t in valid_name_tokens and valid_name_tokens[t] <= thresh:
                    t_cands.update(name_token_idx.get((ctry, t), []))
                    
            c_len = len(t_cands)
            results[thresh]['total_cands'] += c_len
            results[thresh]['cand_counts'].append(c_len)
            
            rec = len(t_cands.intersection(true_m))
            results[thresh]['recovered'] += rec
            
            if thresh == 500:
                missed = true_m - t_cands
                for m in missed:
                    results[500]['misses'].append((s1_id, m))
                    
    for thresh in thresholds:
        res = results[thresh]
        rec_ratio = res['recovered'] / total_true if total_true > 0 else 1.0
        c_arr = np.array(res['cand_counts']) if res['cand_counts'] else np.array([0])
        print(f"\nThreshold: {thresh}")
        print(f"  Recall: {rec_ratio:.4f}")
        print(f"  Total Candidates: {res['total_cands']}")
        print(f"  Candidates per S1 (Mean / Median / P95): {c_arr.mean():.2f} / {np.median(c_arr):.2f} / {np.percentile(c_arr, 95):.2f}")
        
    print("\nE. MISS DIAGNOSIS - 20 SAMPLES (Threshold=500)")
    misses = results[500]['misses']
    random.seed(42)
    sample_misses = random.sample(misses, min(20, len(misses)))
    
    df_s2_idx = df_s2.set_index('entity_id')
    df_s3_idx = df_s3.set_index('entity_id')
    df_s1_idx = df_s1.set_index('entity_id')
    
    print("S1_name | S2/S3_name | S1_addr | S2/S3_addr | S1_country | S2/S3_country")
    for s1_id, s2s3_id in sample_misses:
        s1_row = df_s1_idx.loc[s1_id]
        if s2s3_id in df_s2_idx.index:
            s2s3_row = df_s2_idx.loc[s2s3_id]
        else:
            s2s3_row = df_s3_idx.loc[s2s3_id]
            
        print(f"{s1_row['business_name']} | {s2s3_row['business_name']} | {s1_row['business_address']} | {s2s3_row['business_address']} | {s1_row['country']} | {s2s3_row['country']}")
        print("  | Passes should have caught it: ")
        print("  | Miss cause: ")
        
if __name__ == "__main__":
    analyze()
