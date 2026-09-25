#!/usr/bin/env python
"""
Diagnose blocking recall failure.
- Per-pass contribution analysis
- Missed pair categorization and root-cause quantification
- Name/address similarity distributions
- Recoverability analysis
"""
import os, sys, gc, time, random, re
import collections
import numpy as np
import pandas as pd
from difflib import SequenceMatcher

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.entity_resolution.config import (
    TRAIN_SOURCE1, TRAIN_SOURCE2, TRAIN_SOURCE3, TRAIN_GROUND_TRUTH
)
from src.entity_resolution.blocking import (
    Blocker, normalize_text, strip_legal_suffixes,
    extract_pin_codes, tokenize_address, CHUNK_SIZE
)


def tokenize_address_no_ascii_filter(address):
    """Same as tokenize_address but WITHOUT the isascii() filter."""
    if not isinstance(address, str) or pd.isna(address):
        return set()
    text = address.lower()
    text = re.sub(r'[,\.;/\-\(\)]', ' ', text)
    tokens = text.split()
    valid_tokens = set()
    for t in tokens:
        t = re.sub(r'^[^a-zA-Z0-9\u0900-\u097F]+', '', t)  # keep Devanagari
        t = re.sub(r'[^a-zA-Z0-9\u0900-\u097F]+$', '', t)
        if len(t) >= 2 and not t.isdigit():
            valid_tokens.add(t)
    return valid_tokens


def diagnose():
    t_global = time.time()

    # ================================================================
    # PHASE 1: Load ground truth — same 20k sample
    # ================================================================
    print("=== PHASE 1: Loading ground truth ===", flush=True)
    gt_df = pd.read_csv(TRAIN_GROUND_TRUTH, sep='\t', dtype=str)
    has_match = gt_df['matched_entity_ids'].fillna('').astype(str).str.strip() != ''
    gt_val = gt_df[has_match].sample(n=20_000, random_state=42)
    del gt_df; gc.collect()

    val_s1_ids = set(gt_val['source1_entity_id'])
    y_true = {
        row.source1_entity_id: set(row.matched_entity_ids.split(','))
        for row in gt_val.itertuples(index=False)
    }
    total_true_pairs = sum(len(v) for v in y_true.values())
    print(f"Sample: 20k S1, {total_true_pairs} true pairs", flush=True)
    del gt_val; gc.collect()

    # ================================================================
    # PHASE 2: Build blocker (~25 min)
    # ================================================================
    print("\n=== PHASE 2: Building blocker ===", flush=True)
    t0 = time.time()
    blocker = Blocker(TRAIN_SOURCE2, TRAIN_SOURCE3, rare_freq_threshold=500)
    print(f"Blocker built in {time.time()-t0:.1f}s", flush=True)

    # ================================================================
    # PHASE 3: Load S1 sample rows
    # ================================================================
    df_s1 = pd.read_csv(TRAIN_SOURCE1, sep='\t', dtype=str)
    df_s1_val = df_s1[df_s1['entity_id'].isin(val_s1_ids)].copy()
    del df_s1; gc.collect()
    print(f"S1 rows loaded: {len(df_s1_val)}", flush=True)

    # ================================================================
    # PHASE 4: Per-pass diagnostic + collect missed pairs
    # ================================================================
    print("\n=== PHASE 3: Per-pass diagnostic ===", flush=True)
    pass_names = ['exact_name', 'rare_name_token', 'rare_addr_token', 'pin']
    pass_recovered = {p: 0 for p in pass_names}
    pass_unique = {p: 0 for p in pass_names}
    pass_total_cands = {p: 0 for p in pass_names}

    all_recovered = 0
    all_missed_pairs = []
    s1_data_for_missed = {}

    # Also track: how many S1 entities get 0 candidates from each pass
    pass_zero_s1 = {p: 0 for p in pass_names}
    union_zero_s1 = 0

    for i, row in enumerate(df_s1_val.itertuples(index=False), 1):
        if i % 5000 == 0:
            print(f"  processed {i}/20000", flush=True)

        country = row.country
        norm_name = normalize_text(row.business_name)
        stripped_name = strip_legal_suffixes(norm_name)

        # Pass 1: exact name
        p1 = set(blocker.exact_name_idx.get((country, stripped_name), []))

        # Pass 2: rare name tokens
        p2 = set()
        for token in set(stripped_name.split()):
            p2.update(blocker.rare_token_idx.get((country, token), []))

        # Pass 3: rare address tokens
        p3 = set()
        for token in tokenize_address(row.business_address):
            bucket = blocker.rare_addr_token_idx.get((country, token), [])
            if len(bucket) <= blocker.rare_freq_threshold:
                p3.update(bucket)

        # Pass 4: PIN codes
        p4 = set()
        for pin in extract_pin_codes(row.business_address):
            p4.update(blocker.pin_idx.get((country, pin), []))

        passes = {'exact_name': p1, 'rare_name_token': p2, 'rare_addr_token': p3, 'pin': p4}
        all_cands = p1 | p2 | p3 | p4
        true_m = y_true.get(row.entity_id, set())

        for p_name, p_cands in passes.items():
            found_by_pass = p_cands & true_m
            pass_recovered[p_name] += len(found_by_pass)
            pass_total_cands[p_name] += len(p_cands)
            if len(p_cands) == 0:
                pass_zero_s1[p_name] += 1
            other_cands = set()
            for oname, oset in passes.items():
                if oname != p_name:
                    other_cands |= oset
            unique = found_by_pass - other_cands
            pass_unique[p_name] += len(unique)

        if len(all_cands) == 0:
            union_zero_s1 += 1

        found_total = len(all_cands & true_m)
        all_recovered += found_total

        missed = true_m - all_cands
        if missed:
            s1_data_for_missed[row.entity_id] = {
                'business_name': row.business_name,
                'business_address': row.business_address,
                'country': country,
            }
            for mid in missed:
                all_missed_pairs.append((row.entity_id, mid))

    del df_s1_val; gc.collect()

    # ---- Per-pass report ----
    print("\n" + "=" * 80, flush=True)
    print("PER-PASS DIAGNOSTIC TABLE", flush=True)
    print("=" * 80, flush=True)
    hdr = f"{'Pass':<20} {'Recovered':>10} {'Recall':>8} {'Unique':>8} {'Candidates':>12} {'Zero-S1':>8} {'Prec':>10}"
    print(hdr, flush=True)
    print("-" * 80, flush=True)
    for p in pass_names:
        rec = pass_recovered[p]
        recall = rec / total_true_pairs
        uniq = pass_unique[p]
        cands = pass_total_cands[p]
        prec = rec / cands if cands > 0 else 0
        zero = pass_zero_s1[p]
        print(f"{p:<20} {rec:>10} {recall:>8.4f} {uniq:>8} {cands:>12} {zero:>8} {prec:>10.6f}", flush=True)

    print("-" * 80, flush=True)
    overall_recall = all_recovered / total_true_pairs
    print(f"{'UNION':<20} {all_recovered:>10} {overall_recall:>8.4f} {'':>8} {'':>12} {union_zero_s1:>8}", flush=True)
    print(f"{'MISSED':<20} {len(all_missed_pairs):>10} {len(all_missed_pairs)/total_true_pairs:>8.4f}", flush=True)

    # ================================================================
    # PHASE 5: Look up S2/S3 records for missed pairs
    # ================================================================
    missed_s2s3_ids = set(mid for _, mid in all_missed_pairs)
    print(f"\n=== PHASE 4: Looking up {len(missed_s2s3_ids)} missed S2/S3 records ===", flush=True)
    s2s3_lookup = {}

    for path_label, path in [("S2", TRAIN_SOURCE2), ("S3", TRAIN_SOURCE3)]:
        for chunk in pd.read_csv(path, sep='\t', dtype=str, chunksize=CHUNK_SIZE,
                                  usecols=['entity_id', 'business_name', 'business_address', 'country']):
            mask = chunk['entity_id'].isin(missed_s2s3_ids)
            for row_c in chunk[mask].itertuples(index=False):
                s2s3_lookup[row_c.entity_id] = {
                    'business_name': row_c.business_name,
                    'business_address': row_c.business_address,
                    'country': row_c.country,
                }
            del chunk; gc.collect()
        print(f"  {path_label} done: {len(s2s3_lookup)} found so far", flush=True)

    print(f"Lookup complete: {len(s2s3_lookup)} / {len(missed_s2s3_ids)} missed S2/S3 records", flush=True)

    # ================================================================
    # PHASE 6: Categorize missed pairs
    # ================================================================
    print(f"\n=== PHASE 5: Categorizing {len(all_missed_pairs)} missed pairs ===", flush=True)

    categories = collections.Counter()
    name_sims = []
    addr_sims = []
    token_jaccards = []

    # Sub-categories for deeper analysis
    has_nonascii_addr_count = 0
    has_nonascii_name_count = 0
    nonascii_addr_would_share_tokens = 0
    shared_pin_but_missed = 0
    no_pin_either_side = 0
    has_pin_but_different = 0

    random.seed(42)
    sample_indices = set(random.sample(range(len(all_missed_pairs)), min(100, len(all_missed_pairs))))
    sample_examples = []

    for idx, (s1_id, s2s3_id) in enumerate(all_missed_pairs):
        s1_rec = s1_data_for_missed.get(s1_id, {})
        s2s3_rec = s2s3_lookup.get(s2s3_id, {})

        if not s1_rec or not s2s3_rec:
            categories['lookup_failed'] += 1
            name_sims.append(0)
            addr_sims.append(0)
            token_jaccards.append(0)
            continue

        s1_name = str(s1_rec.get('business_name') or '')
        s2_name = str(s2s3_rec.get('business_name') or '')
        s1_addr = str(s1_rec.get('business_address') or '')
        s2_addr = str(s2s3_rec.get('business_address') or '')
        s1_country = str(s1_rec.get('country') or '')
        s2_country = str(s2s3_rec.get('country') or '')

        # Normalize
        s1_norm = strip_legal_suffixes(normalize_text(s1_name))
        s2_norm = strip_legal_suffixes(normalize_text(s2_name))

        # Name similarity
        name_sim = SequenceMatcher(None, s1_norm, s2_norm).ratio() if s1_norm and s2_norm else 0.0
        name_sims.append(name_sim)

        # Address similarity
        s1_addr_norm = normalize_text(s1_addr)
        s2_addr_norm = normalize_text(s2_addr)
        addr_sim = SequenceMatcher(None, s1_addr_norm, s2_addr_norm).ratio() if s1_addr_norm and s2_addr_norm else 0.0
        addr_sims.append(addr_sim)

        # Token analysis
        s1_tokens = set(s1_norm.split()) if s1_norm else set()
        s2_tokens = set(s2_norm.split()) if s2_norm else set()
        shared_name_tokens = s1_tokens & s2_tokens
        union_tokens = s1_tokens | s2_tokens
        token_jaccard = len(shared_name_tokens) / len(union_tokens) if union_tokens else 0
        token_jaccards.append(token_jaccard)

        # Non-ASCII detection
        na_name = any(not c.isascii() for c in (s1_name + s2_name))
        na_addr = any(not c.isascii() for c in (s1_addr + s2_addr))
        if na_addr:
            has_nonascii_addr_count += 1
            # Check if non-ASCII address tokens would overlap without ascii filter
            s1_na_toks = tokenize_address_no_ascii_filter(s1_addr) - tokenize_address(s1_addr)
            s2_na_toks = tokenize_address_no_ascii_filter(s2_addr) - tokenize_address(s2_addr)
            if s1_na_toks & s2_na_toks:
                nonascii_addr_would_share_tokens += 1
        if na_name:
            has_nonascii_name_count += 1

        # PIN analysis
        s1_pins = set(extract_pin_codes(s1_addr))
        s2_pins = set(extract_pin_codes(s2_addr))
        if s1_pins & s2_pins:
            shared_pin_but_missed += 1
        elif not s1_pins and not s2_pins:
            no_pin_either_side += 1
        elif s1_pins and s2_pins and not (s1_pins & s2_pins):
            has_pin_but_different += 1

        # Categorize by primary failure mode
        if s1_country != s2_country:
            categories['country_mismatch'] += 1
        elif s1_norm == s2_norm and s1_norm:
            categories['BUG_exact_name_match_but_missed'] += 1
        elif name_sim >= 0.9:
            categories['name_very_similar_90+'] += 1
        elif name_sim >= 0.8:
            categories['name_similar_80-90'] += 1
        elif name_sim >= 0.6:
            categories['name_moderate_60-80'] += 1
        elif name_sim >= 0.4:
            categories['name_low_40-60'] += 1
        elif na_name:
            categories['non_ascii_name_very_different'] += 1
        else:
            categories['name_very_different_<40'] += 1

        # Collect sample
        if idx in sample_indices:
            sample_examples.append({
                's1_id': s1_id, 's2s3_id': s2s3_id,
                's1_name': s1_name[:80], 's2_name': s2_name[:80],
                's1_addr': s1_addr[:80], 's2_addr': s2_addr[:80],
                's1_country': s1_country, 's2_country': s2_country,
                'name_sim': name_sim, 'addr_sim': addr_sim,
                'token_jaccard': token_jaccard,
                'shared_pins': len(s1_pins & s2_pins),
                'na_addr': na_addr, 'na_name': na_name,
            })

    # ================================================================
    # REPORTS
    # ================================================================

    # ---- Root cause categories ----
    print("\n" + "=" * 70, flush=True)
    print("ROOT CAUSE CATEGORIES", flush=True)
    print("=" * 70, flush=True)
    print(f"{'Category':<40} {'Count':>8} {'%':>8}", flush=True)
    print("-" * 60, flush=True)
    for cat, count in categories.most_common():
        pct = 100 * count / len(all_missed_pairs)
        print(f"{cat:<40} {count:>8} {pct:>7.1f}%", flush=True)

    # ---- Sub-category analysis ----
    print("\n" + "=" * 70, flush=True)
    print("SUB-CATEGORY ANALYSIS", flush=True)
    print("=" * 70, flush=True)
    n = len(all_missed_pairs)
    print(f"  Non-ASCII in address:                    {has_nonascii_addr_count:>6} ({100*has_nonascii_addr_count/n:.1f}%)", flush=True)
    print(f"    ...would share non-ASCII addr tokens:  {nonascii_addr_would_share_tokens:>6} ({100*nonascii_addr_would_share_tokens/n:.1f}%)", flush=True)
    print(f"  Non-ASCII in name:                       {has_nonascii_name_count:>6} ({100*has_nonascii_name_count/n:.1f}%)", flush=True)
    print(f"  Shared PIN but still missed:             {shared_pin_but_missed:>6} ({100*shared_pin_but_missed/n:.1f}%)", flush=True)
    print(f"  No PIN on either side:                   {no_pin_either_side:>6} ({100*no_pin_either_side/n:.1f}%)", flush=True)
    print(f"  Different PINs:                          {has_pin_but_different:>6} ({100*has_pin_but_different/n:.1f}%)", flush=True)

    # ---- Similarity distributions ----
    ns = np.array(name_sims)
    ads = np.array(addr_sims)
    tj = np.array(token_jaccards)

    print("\n" + "=" * 70, flush=True)
    print("NAME SIMILARITY DISTRIBUTION (missed pairs)", flush=True)
    print("=" * 70, flush=True)
    bins = [0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.01]
    for j in range(len(bins) - 1):
        count = int(np.sum((ns >= bins[j]) & (ns < bins[j + 1])))
        pct = 100 * count / len(ns) if len(ns) > 0 else 0
        print(f"  [{bins[j]:.2f}, {bins[j+1]:.2f}): {count:>6} ({pct:>5.1f}%)", flush=True)
    print(f"  Mean: {ns.mean():.4f}, Median: {np.median(ns):.4f}", flush=True)

    print("\n" + "=" * 70, flush=True)
    print("ADDRESS SIMILARITY DISTRIBUTION (missed pairs)", flush=True)
    print("=" * 70, flush=True)
    for j in range(len(bins) - 1):
        count = int(np.sum((ads >= bins[j]) & (ads < bins[j + 1])))
        pct = 100 * count / len(ads) if len(ads) > 0 else 0
        print(f"  [{bins[j]:.2f}, {bins[j+1]:.2f}): {count:>6} ({pct:>5.1f}%)", flush=True)
    print(f"  Mean: {ads.mean():.4f}, Median: {np.median(ads):.4f}", flush=True)

    print("\n" + "=" * 70, flush=True)
    print("TOKEN JACCARD DISTRIBUTION (missed pairs)", flush=True)
    print("=" * 70, flush=True)
    for j in range(len(bins) - 1):
        count = int(np.sum((tj >= bins[j]) & (tj < bins[j + 1])))
        pct = 100 * count / len(tj) if len(tj) > 0 else 0
        print(f"  [{bins[j]:.2f}, {bins[j+1]:.2f}): {count:>6} ({pct:>5.1f}%)", flush=True)
    print(f"  Mean: {tj.mean():.4f}, Median: {np.median(tj):.4f}", flush=True)

    # ---- Recoverability ----
    print("\n" + "=" * 70, flush=True)
    print("RECOVERABILITY ANALYSIS", flush=True)
    print("=" * 70, flush=True)
    for thresh in [0.5, 0.6, 0.7, 0.8, 0.9]:
        name_rec = int(np.sum(ns >= thresh))
        addr_rec = int(np.sum(ads >= thresh))
        either = int(np.sum((ns >= thresh) | (ads >= thresh)))
        new_recall = (all_recovered + either) / total_true_pairs
        print(f"  Sim >= {thresh:.1f}: name={name_rec:>6} addr={addr_rec:>6} "
              f"either={either:>6} => potential recall={new_recall:.4f}", flush=True)

    # ---- Sample examples ----
    print("\n" + "=" * 70, flush=True)
    print(f"SAMPLE MISSED PAIRS ({len(sample_examples)} examples)", flush=True)
    print("=" * 70, flush=True)
    for ex in sample_examples[:30]:
        print(f"\n  S1 [{ex['s1_country']}]: {ex['s1_id']}", flush=True)
        print(f"    Name:    {ex['s1_name']}", flush=True)
        print(f"    Address: {ex['s1_addr']}", flush=True)
        print(f"  S2/S3 [{ex['s2_country']}]: {ex['s2s3_id']}", flush=True)
        print(f"    Name:    {ex['s2_name']}", flush=True)
        print(f"    Address: {ex['s2_addr']}", flush=True)
        print(f"    NameSim={ex['name_sim']:.3f}  AddrSim={ex['addr_sim']:.3f}  "
              f"TokJac={ex['token_jaccard']:.3f}  PINs={ex['shared_pins']}  "
              f"NonAsciiAddr={ex['na_addr']}  NonAsciiName={ex['na_name']}", flush=True)

    total_time = time.time() - t_global
    print(f"\n{'='*70}", flush=True)
    print(f"Total diagnostic time: {total_time:.1f}s", flush=True)
    print(f"{'='*70}", flush=True)


if __name__ == "__main__":
    diagnose()
