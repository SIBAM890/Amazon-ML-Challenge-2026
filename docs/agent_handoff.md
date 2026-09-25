# Agent Handoff — Amazon ML Challenge 2026 Business Entity Resolution

## 1. Project context

Task: for every Source-1 business record, find all matching Source-2 and
Source-3 records. Zero, one, or many matches per S1 entity. No shared
identifier across sources. Match on business_name, business_address,
country. Ground truth exists for training only.

Metric: macro-averaged F0.5 per Source-1 entity (precision weighted 2x).
Singletons score 1.0 for empty prediction, 0.0 for any predicted match.

Hard rules (never violate):
- student_resource/ is READ-ONLY. Do not modify, rename, or normalize raw files.
- No external lookup: no geocoding, no business registries, no entity-resolution APIs, no internet data.
- country is an OPEN string set. Never hard-code {US, India}. France appears in test.
- Every test S1 entity must appear exactly once in matching_results.tsv.
- matched IDs must be S2-/S3- only, exist in test set, no duplicates.
- candidate_pairs.tsv = the EXACT candidate set fed to the model. Every matched ID must appear there.
- Any pretrained model must be MIT/Apache-2.0 licensed, ≤8B params.
- Run student_resource/utils/validate_submission.py and require PASS before declaring done.

## 2. Machine constraint (non-negotiable)

- 15.3 GB RAM total, typically 3-6 GB free.
- Every stage must be chunked or streamed.
- Never materialize > 2 GB of intermediate data at once.
- Write to parquet between stages.
- Use pd.read_csv(..., chunksize=200_000) for large files.
- Use del + gc.collect() after each chunk.
- Use categorical dtype for country and source columns.
- No .explode() on full-length dataframes. No df.copy() at scale. No df.iterrows() anywhere.

## 3. Verified dataset facts

Paths:
  student_resource/dataset/train/train_source1.tsv
  student_resource/dataset/train/train_source2.tsv
  student_resource/dataset/train/train_source3.tsv
  student_resource/dataset/train/train_ground_truth.tsv
  student_resource/dataset/test/test_source1.tsv
  student_resource/dataset/test/test_source2.tsv
  student_resource/dataset/test/test_source3.tsv

All TSV. Read with sep="\t".

Shapes (verified):
  train_source1:         2,206,821 rows
  train_source2:         5,034,616 rows
  train_source3:         5,285,603 rows
  train_ground_truth:    2,206,821 rows (one per S1)
  S2 + S3 combined:     10,320,219 candidate records

Columns per source file: entity_id, business_name, business_address, country.
Ground truth columns: source1_entity_id, matched_entity_ids.

Ground truth distribution (verified):
  0 matches (singletons):  123,247   (5.6%)
  1 match:                 119,157
  2-5 matches:           1,712,125   (77.6%)
  6+ matches:              252,292
  Total true pairs:      7,638,365
  Median non-empty: 4
  p95 non-empty:    6

Country (verified):
  Train S1:  US 73896 singletons, India 49351 singletons; total US+India.
  Test:      US, India, France (France not in train).
  Cross-country true pairs in training: 0.
  => Same-country hard filter is SAFE. Keep it.

ID rules (verified):
  Non-S2/S3 IDs in matched_entity_ids: 0.

## 4. Token distribution (verified, from train_source2 + train_source3)

business_name:
  S2: 1,025,584 distinct tokens; 17,604,155 total; 2,870 tokens >= 500
  S3: 1,094,895 distinct tokens; 18,634,159 total; 2,725 tokens >= 500
  Top tokens (all legal suffixes / structural): private, llc, limited,
  ltd, inc, &, लिमिटेड, center, प्राइवेट, partners, services, pvt, group,
  corp, co, and, holdings, inc., of, care, (india), associates, llp,
  corporation, service, health, enterprises, industries.

business_address:
  S2: 1,311,166 distinct tokens; 36,677,860 total; 6,302 tokens >= 500
  S3: 1,276,581 distinct tokens; 37,908,121 total; 6,338 tokens >= 500
  Top tokens: road, no, rd, street, st, delhi, maharashtra, no., dr, tx,
  city, floor, new, nagar, ave, delhi, drive, ny, pradesh, nc, plot,
  mumbai, il, west, road, uttar, va, mh, texas, north, unit, york, etc.

Decision: rare_freq_threshold = 500 is defensible. Do not change without
new data.

## 5. Pipeline status

DONE:
- config.py, evaluation.py (macro F0.5 per S1, unit-tested against worked
  example: precision 0.667, recall 1.0, F0.5 = 0.714).
- blocking.py: tokenizers, three name passes + PIN pass, plus address-token
  pass. Blocker constructor signature is now (s2_path, s3_path,
  rare_freq_threshold=500). Internally uses two-pass chunked index build
  (200k-row chunks, no explode, no df.copy, no iterrows, del + gc.collect
  per chunk).
- All call sites of Blocker() updated to pass paths, not DataFrames.
- scratch/token_freq.py and scratch/cross_check.py completed and produced
  the verified numbers above.

IN PROGRESS:
- scratch/time_blocker.py: measuring wall-clock time and peak RSS of the
  chunked Blocker on full 10.3M S2+S3. Run in foreground. Do NOT run in
  background. Report wall-clock seconds and peak RSS in GB when it exits.

PENDING:
- scripts/measure_blocking_recall.py needs two fixes before running:
  (see section 6).
- Then Task 4 = measure blocking recall on a 20k sample of S1 entities
  with >= 1 true match. Target recall >= 0.95. If < 0.90, fix blocking
  before proceeding.
- Then features.py, model.py, inference.py, submission.py, validator.

## 6. Immediate next actions

FIX A — sample filter bug in scripts/measure_blocking_recall.py.
Current code:
    has_match = gt_df['matched_entity_ids'] != ""
This is WRONG because read_csv converts empty fields to NaN, and NaN != ""
is True in pandas, so singletons slip into the sample.
Before fixing, run and paste:
    print("isna count:", gt_df['matched_entity_ids'].isna().sum())
    print("empty-str count:", (gt_df['matched_entity_ids'] == "").sum())
    print("dtypes:", gt_df.dtypes.to_dict())
Then apply one of:
  If empties are NaN:
    has_match = gt_df['matched_entity_ids'].fillna("").astype(str).str.strip() != ""
  If empties are "":
    has_match = gt_df['matched_entity_ids'].astype(str).str.strip() != ""
Confirm filtered count == 2,083,574 (= 2,206,821 - 123,247).

FIX B — verify recall denominator in the same script.
The script must report:
  - number of true matched pairs in the sampled S1 subset
    (sum of len(split) across the 20k sample)
  - number of those pairs present in candidates
  - blocking recall = numerator / denominator
Do NOT compute recall against the full training set. Only against the sample.
Also report mean / median / p95 candidates per S1 entity.

## 7. Behavioral rules for the agent (learned from earlier failures)

- Never claim a script is "running in the background" unless
  Get-Process python* shows a PID. If nothing is running, say so.
- Never substitute an adjacent script for the one requested. If asked to
  run X, run X. If you cannot, say "cannot" — do not produce Y.
- Never change a threshold, a parameter, or a signature to make a metric
  look better. If a metric is bad, diagnose the cause from data first.
- Never narrate numbers you have not measured. If you have not printed a
  value from pandas.shape or len(), do not state it.
- Before any multi-file refactor, list every call site that will break and
  fix them in the same change.
- Run everything in the foreground. Paste raw stdout, not summaries.
- If a script OOMs or is killed by the OS, say so explicitly. Do not
  claim success on a process that died.
- One task at a time. Stop and report after each. Wait for approval.

## 8. What is being measured next and why

Blocking recall = fraction of true matched pairs that appear in the
candidate set produced by blocking. It is a hard ceiling on end-to-end
F0.5: a true match that never becomes a candidate cannot be recovered
by any model. Target >= 0.95 on the 20k sample. If current blocking is
below 0.90, add passes (address tokens already added; next candidates
are character n-gram / TF-IDF retrieval for typos and transliteration)
before touching features or the model.

## 9. Do not do any of the following yet

- Do not start features.py, model.py, inference.py, or submission.py.
- Do not tune rare_freq_threshold.
- Do not modify blocking.py further.
- Do not modify tokenizers.
- Do not re-run token_freq.py or cross_check.py.
- Do not generate any submission file.

Wait for my approval after FIX A and FIX B, then run
scripts/measure_blocking_recall.py in the foreground and report the
full stdout.