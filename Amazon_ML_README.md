# Amazon ML Challenge 2026 — Business Entity Resolution
# Handoff Document

> **Purpose:** Resume work after an IDE quota reset or when switching
> models. Read this entire file first. Then read `docs/agent_spec.md`
> and `docs/agent_handoff.md` for deeper context. Do not invent state
> that is not written here.
>
> **File location:** This file currently lives as `Amazon_ML_README.md`
> in the repo root. It may also be symlinked or copied to
> `docs/handoff.md`. If both exist, they must match.

---

## 0. RESUME HERE — next two messages to send

**Message 1 (verification, do not modify anything):**

```text
Read Amazon_ML_README.md in full, then docs/agent_spec.md and
docs/agent_handoff.md.

You are on branch diag/blocking-recall-0.79.

Current state: blocking.py MAY have been rewritten with Pass B
(char-3gram Jaccard >= 0.2) and union rerank, but the write may not
have completed before the last quota reset. Verification is the first
task.

Do this, in order, without changing any code first:

1. Verify both files parse:
     python -c "import ast; ast.parse(open('src/entity_resolution/blocking.py').read()); print('blocking.py OK')"
     python -c "import ast; ast.parse(open('scripts/measure_blocking_recall.py').read()); print('measure_blocking_recall.py OK')"

2. Print the diff of blocking.py vs. commit 082d673:
     git diff 082d673 -- src/entity_resolution/blocking.py

3. Confirm against Section 7 of this document that every required
   change is present. If anything is missing or the file does not
   parse, list exactly what is missing and wait. If everything is
   present, state so and wait for my go-ahead.

Do NOT run the rebuild yet. Do NOT modify anything. Report items 1-3
and stop.
```

**Message 2 (rebuild, only after Message 1 confirms design is complete):**

```text
Greenlight to run the Pass B rebuild.

Run scripts/measure_blocking_recall.py in the FOREGROUND. Do not
background it.

While it runs, monitor memory every 60 seconds:
  Get-Process -Id <PID> | Select-Object Id, @{N='MemMB';E={[math]::Round($_.WS/1MB,1)}}, CPU
If MemMB crosses 12 GB, kill it and report. We drop chunk size to 50k.

When it finishes, paste the COMPLETE stdout, including:
- Blocker build time in seconds
- Peak RSS if reported
- Candidate generation time in seconds
- Overall recall
- Per-pass recall breakdown
- Mean / median / p95 candidates per S1
- Total candidate pairs

Then save the refreshed parquet:
  data/interim/candidates_20k_sample_v2.parquet

Commit and push:
  git add src/entity_resolution/blocking.py scripts/measure_blocking_recall.py
  git commit -m "feat: Pass B char-3gram blocking at Jaccard>=0.2, union rerank"
  git push origin diag/blocking-recall-0.79

Do NOT modify blocking.py after the run. Do NOT start features.py.
Stop and wait.
```

---

## 1. Problem

For every Source-1 business record, find all matching Source-2 and
Source-3 records. Zero, one, or many matches per S1 entity. No shared
identifier across sources. Match on business_name, business_address,
country. Ground truth exists for training only.

**Metric:** macro-averaged F0.5 per Source-1 entity. Precision weighted
2x over recall. Singletons score 1.0 for empty prediction, 0.0 for any
predicted match.

---

## 2. Hard rules (never violate)

1. `student_resource/` is READ-ONLY. Do not modify, rename, or
   normalize raw files.
2. No external lookup: no geocoding, business registries,
   entity-resolution APIs, or internet data.
3. `country` is an OPEN string set. Never hard-code `{US, India}`.
   France appears in test.
4. Every test S1 entity must appear exactly once in
   `matching_results.tsv`.
5. `matched_entity_ids` must be S2-/S3- only, exist in test set, no
   duplicates within a list, no duplicate S1 rows.
6. `candidate_pairs.tsv` = the EXACT candidate set fed to the model.
   Every ID in `matching_results.tsv` must appear in
   `candidate_pairs.tsv` for that S1.
7. Any pretrained model must be MIT/Apache-2.0 licensed, ≤ 8B params.
8. Run `student_resource/utils/validate_submission.py` and require
   PASS before declaring submission ready.

---

## 3. Machine constraint

- 15.3 GB RAM total, typically 3–6 GB free with IDE running.
- Every stage must be chunked or streamed.
- Never materialize > 2 GB of intermediate data at once.
- Write to parquet between stages.
- Use `pd.read_csv(..., chunksize=200_000)` for large files.
- Use `del` + `gc.collect()` after each chunk.
- Use categorical dtype for country and source columns.
- No `.explode()` on full-length dataframes. No `df.copy()` at scale.
  No `df.iterrows()` anywhere.
- If a design would hold > 100M rows or > 2 GB in RAM, redesign it
  before implementing.

---

## 4. Verified dataset facts

Paths:
```
student_resource/dataset/train/train_source1.tsv
student_resource/dataset/train/train_source2.tsv
student_resource/dataset/train/train_source3.tsv
student_resource/dataset/train/train_ground_truth.tsv
student_resource/dataset/test/test_source1.tsv
student_resource/dataset/test/test_source2.tsv
student_resource/dataset/test/test_source3.tsv
```
All TSV. Read with `sep="\t"`.

Shapes (verified):
```
train_source1:       2,206,821 rows
train_source2:       5,034,616 rows
train_source3:       5,285,603 rows
train_ground_truth:  2,206,821 rows (one per S1)
S2 + S3 combined:   10,320,219 candidate records
```

Columns: `entity_id, business_name, business_address, country`
Ground truth columns: `source1_entity_id, matched_entity_ids`

Ground truth distribution:
```
0 matches (singletons):  123,247   (5.6%)
1 match:                 119,157
2-5 matches:           1,712,125   (77.6%)
6+ matches:              252,292
Total true pairs:      7,638,365
Median non-empty: 4
p95 non-empty:    6
```

Country (verified):
- Train: US + India only.
- Test: US, India, France.
- **Cross-country true pairs in training: 0.**
- Therefore same-country hard filter is SAFE. Keep it.

ID rules (verified): 0 non-S2/S3 IDs in matched_entity_ids.

Token distribution (from train_source2 + train_source3):
```
business_name:
  S2: 1,025,584 distinct, 17.6M total tokens, 2,870 tokens >= 500
  S3: 1,094,895 distinct, 18.6M total tokens, 2,725 tokens >= 500
  Top tokens: legal suffixes and structural words (private, llc,
  limited, ltd, inc, &, center, partners, services, pvt, group,
  corp, co, and, holdings, etc.)

business_address:
  S2: 1,311,166 distinct, 36.7M total tokens, 6,302 tokens >= 500
  S3: 1,276,581 distinct, 37.9M total tokens, 6,338 tokens >= 500
  Top tokens: road, no, rd, street, st, delhi, maharashtra, tx, ny,
  nc, il, plot, mumbai, west, etc.

char-3gram distribution (union of S2+S3 names):
  Total distinct: 80,704
  >= 500: 12,912
  >= 5,000: 4,050
  >= 50,000: 910
  >= 500,000: 15
  Top 3-grams: ent(1.09M), ter(968k), ers(913k), es_(865k), nte(813k),
  ion(806k), ing(785k), ate(681k), com(645k), tio(565k)
  => 3-gram bucket caps are NOT viable. No cap; chunked index.
```

---

## 5. Pipeline status

### DONE AND VERIFIED
- `src/entity_resolution/config.py` — paths to all data files
- `src/entity_resolution/evaluation.py` — macro F0.5 per S1, unit-tested
  against worked example (precision 0.667, recall 1.0, F0.5 ≈ 0.714)
- `src/entity_resolution/normalization.py` — `normalize_script()` using
  `indic-transliteration` (ITRANS scheme), `detect_script()`, plus
  extended `strip_legal_suffixes` covering transliterated forms
  (`praiveta`, `limiTeDa`, etc.)
- Diagnostic parquets in `data/interim/`:
  - `gt_20k_sample.parquet` (source1_entity_id, matched_entity_ids)
  - `candidates_20k_sample.parquet` — from the recall=0.7934 run
  - `missed_classified.parquet` — 100 missed pairs classified
- Diagnostic scripts in `scratch/`:
  - `check_threshold.py`, `check_pass_ab.py`, `check_fpr.py`,
    `diagnose_blocking.py`, `token_freq.py`, `cross_check.py`,
    `time_blocker.py`

### VERIFIED BASELINE (commit 082d673)
- Blocking recall: **0.7934** on a 20k sample of S1 entities with ≥ 1
  true match (73,461 true pairs)
- Mean candidates per S1: 209
- Median: 141
- p95: 639
- Blocker build time: 1272s (~21 min)
- Candidate generation time: 5.5s for 20k S1
- Total candidate pairs produced: 4,184,257

### DIAGNOSTIC FINDINGS (design basis for Pass B)
- Raising the rare-token frequency threshold from 500 → 50000
  degenerates the pass into a no-op / near-full-cross-join. Rejected.
- Threshold sweep on the 20 known missed pairs:
  - threshold=500: 0/20 caught
  - threshold=2000: 11/20 caught at `>=1` shared token, only 1/20
    survives a `>=2` in-band-token gate — that gate design is dead.
- Pass A (shared name tokens ≥ 2, any frequency): TPR 0.40, FPR 0.2475.
  Weak both ways — **dropped, not used**.
- Pass B (char 3-gram):
  - raw shared_3grams ≥ 5: TPR 0.95, FPR 0.4047 — too loose
  - raw shared_3grams ≥ 8: TPR 0.50, FPR 0.2640 — TPR collapses
  - Jaccard ≥ 0.2: TPR 0.95, FPR 0.3369 — **best tradeoff, adopted**
  - Jaccard ≥ 0.3: TPR 0.80, FPR 0.2640 — worse TPR, marginal FPR gain
- 3-gram frequency distribution confirms no viable bucket cap at this
  scale (Section 4) — index must be built uncapped, chunked.
- PIN regex `\b\d{5,6}\b` is AMBIGUOUS (matches street/unit numbers).
  Recommended fix NOT YET APPLIED: 6-digit-only for India (end of
  string or after comma), 5-digit for US/France (end of string or
  after comma), skip PIN for unknown country.
  **Status: flagged, not yet applied. PIN-derived candidates in
  `blocking.py` currently come from the ambiguous regex. Verify before
  trusting PIN pass.**
- Rerank as a pure "top 30 by 3-gram Jaccard" cut was identified as
  risky: it can silently discard true matches found by exact-name or
  PIN that have low text similarity elsewhere. **Fixed by design:**
  rerank is a UNION of (a) top-30 by Jaccard and (b) all
  high-confidence-pass candidates (exact-name, PIN), not a pure cut.
- TPR/FPR numbers came from small samples (20 positive pairs for the
  threshold sweep; ~425 positive-adjacent / 2,125 negative for the
  FPR/TPR table). Treat as directional, not precise. Real in-pipeline
  FPR may be higher than measured since it was computed against random
  negatives, not negatives that already survived a blocking pass.

### IN PROGRESS — exact state at quota reset

Before the reset, the agent had:
- Read the full `blocking.py`
- Written a plan and started rewriting `blocking.py` with Pass B
  (char-3gram, Jaccard ≥ 0.2) and union rerank
- Updated `scripts/measure_blocking_recall.py` to match the new
  `get_candidates()` return type `(Set[str], Dict)`
- Started running `ast.parse` verification on both files
- **The verification command did not complete before the reset.**

**Therefore:**
- The Pass B rebuild has NOT been run.
- No new recall number exists.
- It is UNKNOWN whether the `blocking.py` rewrite fully landed. First
  task of the resuming session is to verify parse and diff. If the
  file parses and diff matches Section 7, proceed. If not, list what
  is missing before doing anything else.

### PENDING — next actions in order
1. Run Message 1 (verification) from Section 0.
2. Confirm blocking.py parses and matches Section 7.
3. If confirmed, run Message 2 (rebuild).
4. Capture: recall, per-pass breakdown, mean/median/p95 candidates,
   blocker build time, candidate gen time, peak RSS.
5. Save `data/interim/candidates_20k_sample_v2.parquet`.
6. Commit and push on `diag/blocking-recall-0.79`.

---

## 6. What the Pass B rebuild is supposed to do

### New Pass B — character 3-gram blocking at Jaccard ≥ 0.2
- Index: `(country, 3gram) -> [entity_ids]`, NO bucket cap
- Query: for each S1, compute 3-grams of normalized (transliterated)
  name. For each candidate appearing in any bucket, count shared
  3-grams and compute Jaccard. Keep if Jaccard ≥ 0.2.

### Rerank — UNION not pure cut
For each S1, final candidate set = UNION of:
- **(a)** Top 30 candidates by 3-gram Jaccard
- **(b)** ALL candidates from high-confidence passes:
  - exact normalized name match
  - PIN exact match

Both sides deduplicated. High-confidence-pass candidates get
guaranteed slots and are not cut by the top-30 rule.

**Rationale:** the top-30 cut could silently discard true matches
found by exact-name or PIN with low 3-gram Jaccard (abbreviations,
reordering). Guaranteed slots prevent that.

### Rationale from diagnostics (see Section 5 for full numbers)
- Pass B at Jaccard ≥ 0.2 is the best TPR/FPR tradeoff found.
- Pass A (shared tokens ≥ 2) is dropped — weak both ways.
- Transliteration normalization strictly improves cross-script pairs,
  no downside on Latin-Latin. Keep.

### Expected outcome
- Overall recall: 0.88–0.93 (not guaranteed 0.90+; small-positive-set
  noise from 20-pair TPR estimate — a lower number is not automatically
  "something broke", check per-pass breakdown first)
- Mean candidates per S1: target ~30 after top-30 cut, but may be
  higher because (b) side is uncapped for entities with very common
  names or PINs
- Peak RSS during Pass B index build: 6–10 GB

### If it OOMs
Drop chunk size from 200k to 50k in `blocking.py`. Same design,
smaller window. Rerun.

### Warning signs in the rebuild output
- Recall lower than the previous 0.79 → rerank is eating matches;
  check per-pass breakdown, likely (b) under-inclusive or (a)'s
  Jaccard cut too tight.
- Mean candidates > 200 after rerank → (b) producing too many slots
  for very common names; consider capping (b) at ~50 per S1.
- Peak RSS > 12 GB during index build → OOM imminent; kill, drop to
  50k chunks.

---

## 7. blocking.py — required content after the Pass B write

Constructor signature:
```python
Blocker(s2_path: str, s3_path: str, rare_freq_threshold: int = 500)
```

Required content:
- `normalize_script` imported from `normalization.py`
- `_count_tokens_in_chunk` applies `normalize_script` to
  `business_name` before `normalize_text + strip_legal_suffixes`
- `_index_chunk` applies same for name; address tokenization unchanged
  (already strips non-ASCII)
- New `tg_idx` dict: `(country, 3gram) -> [entity_ids]`, no cap
- New `_char_trigrams(text)` staticmethod:
  `normalize_script → normalize_text → strip_legal_suffixes → lower →
  remove spaces → padded 3-grams`
- Pass B chunk build in `_index_chunk`: compute 3-grams of
  transliterated name, add to `tg_idx`
- `get_candidates(s1_row)` returns `Tuple[Set[str], Dict]`:
  - first element = final candidate set (reranked, union of a and b)
  - second element = dict mapping candidate_id → which pass contributed
    it (for per-pass recall breakdown)
- Rerank logic:
  ```python
  high_confidence = exact_name_matches | pin_matches
  top30_jaccard = top 30 by 3-gram Jaccard from pass_B matches
  final = high_confidence | top30_jaccard
  ```
- Exact-name pass: unchanged
- PIN pass: unchanged (regex ambiguity flagged in Section 5, NOT yet
  fixed)
- Rare-token and rare-addr-token passes: NOT guaranteed slots in
  rerank (imprecise enough to be cut)
- Pass A (shared tokens ≥ 2, any freq): NOT added

---

## 8. Behavioral rules for the agent (learned from failures)

- **Never** claim a script is "running in the background" unless
  `Get-Process python*` shows a PID with that script in its command
  line. Verify before claiming.
- **Never** substitute an adjacent script for the one requested. If
  asked to run X, run X. If you cannot, say "cannot."
- **Never** change a threshold, parameter, or signature to make a
  metric look better. If a metric is bad, diagnose the cause from data
  first.
- **Never** narrate numbers you have not measured. Every number must
  come from a `print()`, `pandas.shape`, or file listing.
- **Before** any multi-file refactor, list every call site that will
  break and fix them in the same change.
- **Run everything in the foreground.** Paste raw stdout, not
  summaries.
- **If a script OOMs or is killed by the OS**, say so explicitly. Do
  not claim success on a process that died.
- **One task at a time.** Stop and report after each. Wait for
  approval.
- **If a print crashes on Unicode**, add
  `sys.stdout.reconfigure(encoding="utf-8")` rather than "fixing"
  the data.

---

## 9. Verified numbers to reference

| Metric | Value |
|---|---|
| Total S1 (train) | 2,206,821 |
| S2 + S3 (train) | 10,320,219 |
| Total true pairs | 7,638,365 |
| Singletons | 123,247 (5.6%) |
| Median matches per S1 | 4 |
| Cross-country true pairs | 0 |
| Baseline recall (commit 082d673) | 0.7934 |
| Baseline mean candidates / S1 | 209 |
| Baseline median candidates / S1 | 141 |
| Baseline p95 candidates / S1 | 639 |
| Baseline blocker build | 1272s |
| Baseline candidate gen time | 5.5s / 20k S1 |
| Threshold sweep: 500 catches (of 20 known misses) | 0/20 |
| Threshold sweep: 2000 catches `>=1` (of 20) | 11/20 |
| Threshold sweep: 2000 catches `>=2` gate (of 20) | 1/20 |
| Pass A TPR / FPR (shared tokens ≥2) | 0.40 / 0.2475 |
| Pass B TPR / FPR at raw shared_3grams ≥5 | 0.95 / 0.4047 |
| Pass B TPR / FPR at Jaccard ≥ 0.2 (adopted) | 0.95 / 0.3369 |
| Pass B TPR / FPR at Jaccard ≥ 0.3 | 0.80 / 0.2640 |
| Distinct 3-grams (union S2+S3) | 80,704 |
| 3-grams appearing ≥ 500 | 12,912 |
| 3-grams appearing ≥ 50,000 | 910 |

---

## 10. Git state

- Branch `main`: commit `082d673` — first verified blocking recall
  0.7934. Pushed to `origin/main`.
- Branch `diag/blocking-recall-0.79`: diagnostic work on the 0.7934
  baseline, includes transliteration, Pass B prep, diagnostic scripts.
  The uncommitted Pass B rewrite of `blocking.py` may be present here.
  Check with `git status` and `git diff` before starting.
- Untracked (gitignored, stays local):
  - `student_resource/` (challenge data — DO NOT commit)
  - `data/interim/*.parquet` (regenerable)
  - `scratch/` (diagnostic scripts — can be committed if desired)

**Before running the Pass B rebuild, commit current state:**
```powershell
git status
git add src/entity_resolution/normalization.py src/entity_resolution/blocking.py scripts/measure_blocking_recall.py
git add scratch/check_threshold.py scratch/check_pass_ab.py scratch/check_fpr.py scratch/diagnose_blocking.py
git commit -m "diag: Pass B at jaccard>=0.2, union rerank design validated, FPR=0.34 TPR=0.95"
git push origin diag/blocking-recall-0.79
```

If `git status` shows only the working tree changes you expect, commit.
If it shows unexpected files (e.g., someone modified `student_resource/`),
STOP and report before committing.

---

## 11. Decision tree after the Pass B rebuild

| Outcome | Next action |
|---|---|
| Recall ≥ 0.90, mean cand ≤ 80 | **Proceed to features** (Step 5) |
| Recall ≥ 0.90, mean cand > 200 | Tighten top-30 cut to top-15, rerun |
| Recall 0.85–0.89 | Add AND variant of Pass A+B, or adjust Jaccard threshold, rerun |
| Recall < 0.85 | Diagnose per-pass breakdown; likely Pass B threshold too strict or rerank eating matches via union bug |
| OOM during index build | Drop chunk size to 50k, rerun |

---

## 12. Do NOT do any of the following yet

- Do NOT start `features.py`, `model.py`, `inference.py`, or
  `submission.py`.
- Do NOT tune `rare_freq_threshold`.
- Do NOT modify tokenizers.
- Do NOT re-run `token_freq.py` or `cross_check.py` (their outputs are
  captured in Section 4).
- Do NOT generate any submission file.
- Do NOT allow the agent to touch `blocking.py` after the rebuild
  starts. One change at a time.
- Do NOT "fix" the PIN regex during this rebuild. It stays as-is for
  now; fixing it is a separate change with its own rebuild and
  validation.

---

## 13. Contact points

- Full technical spec: `docs/agent_spec.md`
- Agent behavioral context: `docs/agent_handoff.md`
- Challenge problem statement: `student_resource/README.md`
- Validator: `student_resource/utils/validate_submission.py`

**The official challenge files always take precedence over any
assumption in this handoff document.**