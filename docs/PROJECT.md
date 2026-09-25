# Amazon ML Challenge 2026 — Business Entity Resolution
## Consolidated Project State & Agent Guide

**Single source of truth.** If anything here conflicts with an older
file, this file wins. If it conflicts with the official problem
statement in `student_resource/README.md`, the problem statement wins.

**How to use this file:**
- New session / new model / quota reset: read this top to bottom first.
- Every agent prompt in §14 is ready to paste.
- §7 through §10 contain decisions and rationales. Do not re-litigate
  them without new data.
- §13 lists experiments already tried and failed. Do not re-run them.

---

## 1. The problem

Three independent business-record sources, no shared identifier.

- **Source 1**: deduplicated reference. 2,206,821 rows.
- **Source 2**: noisy. 5,034,616 rows.
- **Source 3**: noisy. 5,285,603 rows.

For every Source-1 record, find all matching Source-2 and Source-3
records. Zero, one, or many matches per S1. Match on `business_name`,
`business_address`, `country` only.

**Metric**: macro-averaged F0.5 per Source-1 entity.
```
F0.5 = (1.25 × P × R) / (0.25 × P + R)
```
Precision weighted 2× over recall. Singletons count: empty prediction
on a true singleton = 1.0, any prediction = 0.0.

**Deliverables**:
- `output/matching_results.tsv` — the only file scored on the leaderboard.
- `output/candidate_pairs.tsv` — the exact pre-model candidate set. Not
  scored, used to audit blocking. Every matched ID must appear here.

---

## 2. Hard rules — never violate

1. `student_resource/` is READ-ONLY. No edits, renames, or normalization
   in place.
2. **No external lookup**: no geocoding, no business registries, no
   entity-resolution APIs, no internet data. Disqualification-level.
3. `country` is an **open** string set. Never hard-code `{US, India}`.
   Test includes France, absent from training. Every test S1 entity
   must appear in the submission regardless of country.
4. Every test S1 entity appears **exactly once** in
   `matching_results.tsv`.
5. `matched_entity_ids`:
   - Only `S2-`/`S3-` IDs.
   - Must exist in the test set.
   - No duplicates within a list.
   - Empty string for singletons (not `None`, not `NaN`).
6. `candidate_pairs.tsv`: same rules; every matched ID must appear here
   for that S1.
7. Pretrained models: MIT or Apache-2.0 license, ≤8B parameters.
8. Before submission is declared ready, run
   `student_resource/utils/validate_submission.py` and require **PASS**.

---

## 3. Machine constraints (non-negotiable)

- 15.3 GB RAM total, 3–6 GB free with IDE running.
- Every stage chunked or streamed. Never materialize >2 GB intermediate
  at once.
- Use `pd.read_csv(..., chunksize=200_000)` for large files.
- `del` + `gc.collect()` after each chunk.
- Categorical dtype for `country` and source columns.
- No `.explode()` on full-length frames. No `df.copy()` at scale. No
  `df.iterrows()` anywhere in the codebase.
- No `import pandas` loaded DataFrames longer than necessary — always
  chunk.

---

## 4. Verified dataset facts

**Shapes** (verified via `pandas.shape`):
```
train_source1:       2,206,821
train_source2:       5,034,616
train_source3:       5,285,603
train_ground_truth:  2,206,821   (1 row per S1)
S2+S3 combined:     10,320,219
```

**Columns**: `entity_id, business_name, business_address, country`.
Ground truth: `source1_entity_id, matched_entity_ids`.

**Ground truth distribution**:
```
0 matches (singletons):  123,247   (5.6%)
1 match:                 119,157
2-5 matches:           1,712,125   (77.6%)
6+ matches:              252,292
Total true pairs:      7,638,365
Median non-empty: 4
p95 non-empty:    6
```

**Country**:
- Train: US, India only.
- Test: US, India, France.
- **Cross-country true pairs in training: 0.** Same-country filter is
  safe. Keep it as a hard filter everywhere.

**ID rules**: 0 non-S2/S3 IDs appear in `matched_entity_ids`. Clean.

**Token distribution** (S2+S3 union):
```
names:     1.03M distinct, 2,870 tokens ≥ 500 (all legal suffixes)
addresses: 1.31M distinct, 6,302 tokens ≥ 500 (mostly structural)
3-grams:   80,704 distinct; 12,912 ≥ 500; 4,050 ≥ 5,000; 910 ≥ 50,000
  top: ent(1.09M), ter(968k), ers(913k), es_(865k), nte(813k),
       ion(806k), ing(785k), ate(681k), com(645k), tio(565k)
```

---

## 5. Current pipeline state — LOCKED

### Blocking recall (validation)
```
0.7977  on 20k S1 sample with ≥1 true match (73,461 true pairs)
```

Per-pass attribution (`first-attribution`, on the 20k sample):
```
exact_name      0.4045   (29,718 pairs)
rare_token      0.1819   (13,361)
rare_addr_token 0.1829   (13,439)
pin             0.0273   ( 2,009)
pass_b          0.0010   (    75)   ← negligible
missed_by_all   0.2023   (14,859)   ← architectural ceiling
```

Candidate-set size (validation):
```
Mean   210.6 per S1
Median 142
p95    642
Total  4,211,765 pairs (20k sample)
→ extrapolated test set: ~460M pairs
```

### Rerank: REMOVED
Pure union of all passes. **No** top-30 cut. **No** candidate-count cap
inside blocking. Precision is enforced downstream.

Reason: an earlier version cut candidates to
`top-30 Pass-B Jaccard ∪ exact_name ∪ PIN`. It collapsed recall from
0.7977 to 0.4362 because rare_token and rare_addr_token candidates were
not guaranteed slots. Lesson: **blocking must be generous; the
classifier handles precision.**

### Pass B (char 3-gram)
- Jaccard ≥ 0.2, index bucket cap = freq ≤ 5000.
- Contributes 0.0010 recall. Weak but not removed — it's cheap at cap
  5000 (51s gen for 20k S1).
- Cap tested at 20000: recall 0.8038 vs 0.7977 (+0.6%), gen time 399s vs
  51s (8×). **Reverted to 5000.** Do not raise again without new data.

### Recall ceiling
~0.80 on the non-singleton sample. 20.2% of true pairs are missed by
every pass. No classifier can recover them. Full macro-F0.5 will be
higher than 0.80 because ~5.6% of S1 are singletons scoring 1.0 for
correct empty predictions.

### Files
```
src/entity_resolution/
  config.py               paths, seeds
  normalization.py        normalize_script (ITRANS transliteration),
                          detect_script, strip_legal_suffixes
                          (incl. transliterated forms)
  blocking.py             Blocker class, chunked 2-pass index build,
                          4 passes + Pass B, pure union
  evaluation.py           macro F0.5 per S1 (unit-tested against
                          worked example: 0.714)
scripts/
  measure_blocking_recall.py    validated
  run_baselines.py              exists, baselines useless on matched sample
data/interim/
  gt_20k_sample.parquet         20000 S1, matched_entity_ids
  candidates_20k_sample.parquet from recall=0.7934 run (old)
  candidates_20k_sample_v2.parquet   from recall=0.7977 run (current)
  missed_classified.parquet     100 missed pairs classified
```

Constructor:
```python
Blocker(s2_path: str, s3_path: str, rare_freq_threshold: int = 500)
```

`get_candidates(s1_row)` returns `(Set[str], Dict[str, str])` — the
candidate set and a pass-attribution dict for diagnostics.

---

## 6. Environment

```
Python 3.13
pandas, numpy, scipy, scikit-learn, rapidfuzz,
indic-transliteration, pyarrow, psutil, pytest
```

Model constraint: MIT/Apache-2.0, ≤8B params. Log license + parameter
count in `docs/methodology.md` if you introduce any pretrained model.

---

## 7. Decisions locked (do not re-litigate)

| Decision | Rationale |
|---|---|
| Same-country hard filter | 0 cross-country true pairs in training. Verified. |
| `rare_freq_threshold = 500` | Token distribution shows 2,870 name tokens ≥ 500, all legal suffixes. Lower explodes blocks, higher drops signal. |
| Transliteration via ITRANS | Strict improvement on cross-script misses (5/20 improved, 0 harmed). |
| Pure union, no rerank | Earlier top-30 cut collapsed recall 0.79→0.44. |
| Pass B at Jaccard ≥ 0.2 | Best TPR/FPR on 20-pair calibration (TPR 0.95, FPR 0.337). |
| Pass B bucket cap = 5,000 | Prevents unbounded enumeration. 20,000 tested → not worth the 8× compute. |
| Real Jaccard formula | `shared / (|A| + |B| − shared)`. The approximation `shared / (|A| + shared)` passed 1.19× more pairs at same threshold. |
| Chunked index build (200k rows) | 15.3 GB RAM; full-frame explode OOMs. |
| No `iterrows` anywhere | 10.3M rows × iterrows = multi-hour. Vectorize or `itertuples`. |

---

## 8. Blocking design (final)

**Pass 1 — exact normalized name**
Key: `(country, stripped_name)`. Normalize: `normalize_script →
normalize_text → strip_legal_suffixes → lower`. Strips legal suffixes
including transliterated forms (`praiveta`, `limiTeDa`, etc.).

**Pass 2 — rare name token**
Key: `(country, token)`. Token appears ≤500 times in S2+S3 union.

**Pass 2b — rare address token**
Key: `(country, addr_token)`. Purely numeric tokens dropped. Token
appears ≤500 times. Bucket cap 500.

**Pass 3 — PIN**
Regex `\b\d{5,6}\b` (known ambiguous — matches street numbers, unit
numbers). Flagged but not fixed. Do not fix without a rebuild plan.

**Pass B — char 3-gram**
Jaccard on padded 3-grams of normalized name. Threshold ≥ 0.2.
Index bucket cap = freq ≤ 5000. Contributes 0.001 recall.

**Union** — final = union of all passes. No cut, no cap, no rerank.

---

## 9. Diagnostic history (why we know what we know)

- **Pass A test** (shared name tokens ≥2, any frequency): TPR 0.40,
  FPR 0.2475. Weak both ways. **Rejected.**
- **Threshold 500 → 2000 sweep** on 20 known misses: 11/20 at ≥1, only
  1/20 at ≥2 gate. **The gate design is dead.**
- **Pass B raw shared 3-grams sweep**:
  - ≥5: TPR 0.95, FPR 0.4047 (too loose)
  - ≥8: TPR 0.50, FPR 0.2640 (TPR collapses)
  - **Jaccard ≥ 0.2: TPR 0.95, FPR 0.337** (adopted)
  - Jaccard ≥ 0.3: TPR 0.80, FPR 0.2640 (worse TPR)
- **Real vs approx Jaccard calibration** on 500 random pairs: approx
  passes 272, real passes 228. Ratio 1.1930. Approx was too permissive;
  fix was necessary.
- **Cap=20000 test**: recall 0.8038 vs 0.7977 at cap=5000. 8× compute
  for +0.6%. Reverted.
- **Rerank top-30 test**: recall 0.4362. Reverted to pure union.
- **Miss classifications** from 100 sampled pairs: 25% transliteration,
  20% typos, 20% concatenation/domain suffix, 25% all-common tokens,
  10% other.

---

## 10. Known issues (flagged, not fixed)

- **PIN regex ambiguous.** `\b\d{5,6}\b` matches street numbers and
  unit numbers, not just postal codes. Fix requires a country-aware
  extractor and a rebuild. Do not fix casually.
- **Pass B is weak** (0.001 recall). Contributes little.
- **20.2% of true pairs missed by all passes.** This is the architectural
  ceiling. Fixing requires a different blocking approach (e.g. LLM
  embeddings, learned retrieval, TF-IDF ANN), which is out of scope for
  the remaining budget.

---

## 11. Next steps — in order

### Step A — Retrieval stage (before features)
Cut candidate count from ~210/S1 to ~50/S1 for feature computation.
**Must use union pattern** (this is the lesson from the rerank bug):

```python
final = (all exact_name candidates)
      ∪ (all PIN candidates)
      ∪ (top-K by cheap char-3-gram Jaccard on name+address,
         from the remaining pool)
```

`cheap_similarity = jaccard_3gram(name+addr) + 0.2 * country_match`.
Test K ∈ {25, 50, 100} on the 20k sample. Target: K=50 loses ≤2 points
of recall vs 0.7977.

Write the retrieval step as a new function `retrieve()` in
`src/entity_resolution/retrieval.py`. Do NOT modify `blocking.py`.

### Step B — Features
`src/entity_resolution/features.py` with 12–15 pairwise features:
- name: exact, Levenshtein ratio, Jaro-Winkler, token Jaccard,
  char-3-gram Jaccard, TF-IDF cosine
- address: token Jaccard, Levenshtein ratio, TF-IDF cosine,
  city-token match, PIN exact (country-aware), street-number match
- combined: country exact, name-length diff, address-length diff

Unit-test on hand-built pairs. Deterministic. Handle NaN → 0.0.

### Step C — Classifier
`src/entity_resolution/model.py` — `PairMatcher` wrapping LogisticRegression
with `class_weight="balanced"`. Optionally LightGBM later.

`scripts/train_model.py`:
- Load retrieval output parquet.
- Label from ground truth.
- **Split at the S1-entity level** (MD5 hash of `s1_id` → train/val).
  Never split at pair level.
- Fit LogReg. Threshold sweep 0.30–0.95 step 0.05.
- Choose threshold maximizing macro F0.5 per S1 on validation.
- Save model + threshold.

### Step D — Inference
`src/entity_resolution/inference.py`:
- Chunk test S1 by 10k. For each chunk:
  - Block → retrieve → features → score → threshold.
  - Collect per-S1 predictions.
- Write `output/matching_results.tsv` (one row per test S1, empty for
  singletons) and `output/candidate_pairs.tsv` (retrieval output, NOT
  blocking output — the pre-model set).

### Step E — Validate + submit
```
python student_resource/utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir student_resource/dataset/test
```
Require PASS. Then upload `matching_results.tsv` only.

### Timing (with ~42 hours remaining)
- Retrieval test: 1h
- Features + unit tests: 2h
- Train + threshold: 1h
- Inference at scale (2.2M S1): 2–4h
- Validate + upload: 1h
- Buffer: 4h

Do not spend more time on blocking. The 0.7977 ceiling is locked.

---

## 12. Reference numbers

| Metric | Value |
|---|---|
| Train S1 | 2,206,821 |
| Train S2+S3 | 10,320,219 |
| True pairs | 7,638,365 |
| Singletons | 123,247 (5.6%) |
| Median matches/S1 | 4 |
| Cross-country true pairs | 0 |
| **Blocking recall** | **0.7977** |
| Mean candidates/S1 | 210.6 |
| Median candidates/S1 | 142 |
| p95 candidates/S1 | 642 |
| Blocker build (cap=5000) | ~1370–1830s |
| Candidate gen (20k S1) | 51s |
| Pass B recall contribution | 0.0010 |
| Missed by all passes | 0.2023 |
| Distinct 3-grams | 80,704 |
| 3-grams ≥ 5,000 | 4,050 |

---

## 13. Experiments already tried — DO NOT REPEAT

- Raising `rare_freq_threshold` to 2000, 5000, or 50000: worse.
- First-token blocking: explodes candidate set.
- Pass A (shared name tokens ≥2): TPR 0.40 / FPR 0.25. Rejected.
- Rerank with top-30 Pass-B Jaccard cut: collapses recall. Rejected.
- Pass B at raw shared 3-grams ≥5 or ≥8: worse than Jaccard ≥0.2.
- Pass B cap=20000: +0.6% recall, 8× compute. Reverted.
- `df.iterrows()` anywhere: multi-hour hang on 10M rows.
- Full-frame `.explode()`: OOM on 15.3 GB.
- Approximation Jaccard `shared/(|A|+shared)`: 1.19× permissive vs real.
- All-singletons baseline on the matched 20k sample: 0.0 (tautology).
- Random-4 baseline: 0.0.

---

## 14. Behavioral rules for agents

These exist because they were violated and cost real time.

- **Never claim a script is running** unless `Get-Process python*` shows
  a PID with that script in its CommandLine. Verify before claiming.
- **Never substitute a script** for the one requested. If asked to run
  X, run X. If you cannot, say "cannot."
- **Never change a threshold, parameter, or signature** to improve a
  metric. Diagnose from data first.
- **Never narrate numbers you have not printed.** Every number must come
  from `print()`, `.shape`, or a file listing.
- **Before a multi-file refactor**, list every call site that will break
  and fix them in the same change.
- **Run everything in the foreground.** Paste raw stdout, not summaries.
- **If a script OOMs or is killed by the OS**, say so explicitly. Do not
  claim success on a process that died.
- **One task at a time.** Stop and report. Wait for approval.
- **If `print()` crashes on Unicode**, add
  `sys.stdout.reconfigure(encoding="utf-8")` — do not modify data.
- **Do not run the rebuild through the agent's tool wrapper.** It has a
  60-minute cap and the blocker build exceeds it. Always run long
  scripts manually in a plain PowerShell window.

---

## 15. Git state

- Branch `main`: `082d673` — first verified blocking recall 0.7934.
- Branch `diag/blocking-recall-0.79`: current work. Contains the
  transliteration, Pass B, pure-union fix, cap=5000, chunked index,
  diagnostics scripts.
- Untracked (gitignored, stays local):
  - `student_resource/` — challenge data. Never commit.
  - `data/interim/*.parquet` — regenerable.
  - `scratch/` — diagnostics. Committable if desired.

Commit convention:
```
feat(blocking): <what>
fix(blocking): <what> (was X, now Y)
docs: <what>
diag: <what>
```

Commit after each critical-path step. Do not squash.

---

## 16. Do NOT do

- Do not modify `student_resource/`.
- Do not modify `normalization.py` tokenizers.
- Do not tune `rare_freq_threshold`.
- Do not raise Pass B bucket cap above 5000.
- Do not add a rerank or top-K cut **inside** `blocking.py`.
- Do not add a cut inside `retrieve()` that drops exact_name or PIN
  candidates. Those are guaranteed slots.
- Do not use `.explode()`, `.copy()`, or `.iterrows()` on full frames.
- Do not run rebuilds through the agent wrapper.
- Do not upload anything before the official validator returns PASS.
- Do not re-run experiments in §13.

---

## 17. First message for a fresh session

Paste this verbatim:

```text
Read PROJECT.md in full. Then read student_resource/README.md.

You are on branch diag/blocking-recall-0.79.

Current state (do not re-derive):
- Blocking recall locked at 0.7977 on 20k non-singleton sample.
- Rerank removed. Pure union of all passes.
- Pass B at Jaccard>=0.2, bucket cap freq<=5000.
- Next task is Step A in §11: retrieval stage (union pattern) to cut
  candidates from ~210/S1 to ~50/S1 before features.

Do this, without modifying anything:

1. Confirm git branch and status:
     git branch
     git status
2. Confirm the following parse:
     python -c "import ast; ast.parse(open('src/entity_resolution/blocking.py').read()); print('ok')"
     python -c "import ast; ast.parse(open('scripts/measure_blocking_recall.py').read()); print('ok')"
3. Report §11 Step A as your plan. Do not write code yet.
Stop and wait for approval.
```

Then greenlight Step A with a scoped instruction (retrieval only, not
features).

---

*Official challenge files in `student_resource/` always take precedence
over anything in this document.*
