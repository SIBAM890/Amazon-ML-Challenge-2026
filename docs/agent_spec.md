# Amazon ML Challenge 2026 — Business Entity Resolution

---

## HARD CONSTRAINT — MACHINE MEMORY

Target machine has 15.3 GB RAM total, typically 3-6 GB free after IDE.
Every stage must be chunked or streamed.
Never materialize >2 GB of intermediate data at once.
Write to parquet between stages.
If a design decision would hold >100M rows or >2 GB in RAM, redesign it
before implementing.
Use `pd.read_csv(chunksize=...)` for large files.
Use `del + gc.collect()` after each chunk.
Use categorical dtype for `country`, `entity_id` prefix, source columns.

---
## AI Coding Agent Specification (Antigravity-ready)

> Two layers. **Layer 1** is what you paste into your agent as the persistent
> project instruction — keep it short, the agent will actually follow it.
> **Layer 2** is the full reference spec — the agent reads it when it needs
> detail, but does not need to hold all of it in working context at once.

---

## LAYER 1 — Persistent Agent Instruction

*(Paste this whole block into Antigravity as the system/project instruction.)*

```text
You are the ML engineering agent for Amazon ML Challenge 2026 — Business
Entity Resolution. Full reference spec: docs/agent_spec.md (this file, Layer 2).

HARD RULES (never violate, no exceptions):
1. Treat student_resource/ as READ-ONLY. Never edit or "clean" the raw
   dataset files in place.
2. No external business/entity lookup, no geocoding APIs, no commercial
   entity-resolution services, no internet data augmentation of any kind.
   Disqualification-level rule.
3. Treat `country` as an open set of string labels. Never hard-code
   ["US", "India"]. The test set adds "France", which does not appear in
   training. Every test Source-1 entity must appear in the submission
   regardless of country.
4. Metric is macro-averaged F0.5 per Source-1 entity (precision weighted
   2x over recall). False merges are penalized harder than missed matches.
5. Singletons: a Source-1 entity with no true match scores 1.0 for an empty
   prediction, 0.0 for any predicted match. Do not force a match onto a
   low-confidence entity just to avoid an empty row.
6. candidate_pairs.tsv = the EXACT candidate set fed to the matching model
   at inference time — the last stage of blocking/filtering, not an early
   blocking dump you later narrow. Every ID in matching_results.tsv MUST
   appear in candidate_pairs.tsv for that source1_entity_id.
7. matching_results.tsv rules: every test Source-1 ID appears exactly once;
   matched_entity_ids contains only S2-/S3- prefixed IDs that exist in the
   test set; no duplicate IDs within a list; no duplicate source1_entity_id
   rows; empty string (not "None"/"NaN") for singletons.
8. Any pretrained model used (embeddings, sentence-transformers, etc.) must
   be MIT or Apache-2.0 licensed and ≤8B parameters. Log the model name,
   license, and parameter count in docs/methodology.md the moment you
   introduce it.
9. Before declaring the submission ready, run the OFFICIAL validator
   (student_resource/utils/validate_submission.py — confirm the actual path
   during recon, this is the authoritative location) and require a PASS.

CRITICAL PATH — do these seven tasks in order. Stop after each one and
report what you found/built before moving on. Do not skip ahead.

1. RECON (read-only). Inspect README, the validator script, and all TSVs
   (train + test, all 3 sources + ground truth). Report: column schemas,
   row counts per file, distinct country values per split, ground-truth
   structure, and the singleton vs. multi-match distribution in training.
   Make no edits in this step.

2. SCAFFOLD. Minimal src/entity_resolution/ package + scripts/ folder.
   A single config.py pointing at the real student_resource/ paths found
   in recon. No empty placeholder modules beyond what step 3+ needs.

3. EVALUATOR. evaluation.py implementing macro F0.5 per Source-1 entity,
   exactly matching the formula and per-entity averaging described in the
   problem statement. Unit-test it against the worked example in the
   problem statement (S1-00001 → precision 0.667, recall 1.0, F0.5 ≈ 0.714).
   Also compute two baselines on your validation split and report them:
   (a) predict all-singletons, (b) a naive random/majority predictor.
   These are your floor — any real model must beat both.

4. BLOCKING. blocking.py with multiple candidate-generation passes, unioned
   and deduplicated:
     - normalized-name exact/near match (strip legal suffixes: Pvt/Ltd/
       Corp/Inc/Private/Limited, lowercase, strip punctuation)
     - rare name token match (token frequency below a threshold — NOT
       "first token", which explodes on generic words like "The"/"Global"/
       "ABC"; reserve "first token" for a feature, not a blocking key)
     - address/locality token match (city, PIN/postal code if present)
     - same-country pre-filter applied everywhere (cross-country candidates
       are near-impossible true matches and this cuts search space safely)
   Build measure_blocking_recall.py, run it on a held-out validation split,
   and report the number. TARGET: blocking recall ≥ 0.95. If it's below
   0.90, stop — do not proceed to features/model until blocking is fixed.

5. FEATURES. features.py producing pairwise features per candidate pair:
   name similarity (Levenshtein ratio, Jaro-Winkler, token Jaccard, TF-IDF
   cosine on char n-grams, suffix-stripped exact-match flag), address
   similarity (token Jaccard, city match flag, PIN/postal exact-match flag,
   street-number match), and country exact-match flag. Unit-test on a
   handful of hand-built pairs with known expected similarity direction.

6. MODEL. model.py + train_model.py. Start with Logistic Regression (or an
   equally simple, fast, licensable baseline) on the engineered features.
   Sweep the decision threshold and select the one that maximizes macro
   F0.5-per-entity on the validation split (not global accuracy, not the
   default 0.5 — F0.5's precision weighting typically pushes the optimal
   threshold higher than accuracy-optimal). On ties within 0.001, prefer
   the HIGHER threshold — favors precision, consistent with F0.5. Report
   validation F0.5 against the two baselines from step 3.

7. INFERENCE + SUBMISSION. inference.py + submission.py generating both
   matching_results.tsv and candidate_pairs.tsv for the real test set. Run
   the official validator. Require PASS before reporting this step done.

NICE-TO-HAVE — only after all seven critical-path steps are complete and
validator-passing:
   - LightGBM/XGBoost vs. Logistic Regression comparison
   - TF-IDF + approximate-nearest-neighbor retrieval as an extra blocking
     pass to push recall further
   - Per-entity or per-country adaptive thresholds
   - 02_error_analysis.ipynb (cases where blocking recall found the true
     match but the model/threshold missed it, vs. cases blocking never
     found it — these need different fixes)
   - Full docs/methodology.md write-up (methodology, blocking strategy,
     model + features, results, limitations — no page limit required)

WORKING STYLE for every task above:
   - Before changing code: state what you're about to do and why in one
     or two lines.
   - Make the smallest correct change that accomplishes the step — do not
     refactor unrelated code while doing it.
   - After changing code: run the relevant tests/scripts and report the
     ACTUAL numeric output (recall, F0.5, row counts, validator result),
     not a description of what you expect it to show.
   - Commit after each critical-path step with a descriptive message
     (e.g. "step 4: multi-pass blocking, recall 0.962 on val split").
     Do not squash commits.
```

---

## LAYER 2 — Full Reference Spec

### 2. Problem summary

Given noisy business records from three independent sources (Source 1 =
deduplicated reference, Source 2 and Source 3 = noisy candidates), find
every Source 2/3 record that refers to the same real-world business as each
Source 1 record. A Source 1 entity may have zero, one, or many true
matches. No shared identifier exists across sources — matching is done
purely from `business_name`, `business_address`, and `country`.

### 3. Non-negotiable rules (source-of-truth traps)

These four are explicit in the official problem statement and are the ones
most likely to silently cost score or trigger disqualification if missed:

| # | Rule | Why it matters |
|---|---|---|
| A | `candidate_pairs.tsv` must be the exact, final pre-model candidate set — not an early/intermediate blocking dump. Every matched ID must appear in it. | The validator explicitly checks matches ⊆ candidates; mismatch signals a pipeline bug and is flagged. |
| B | `country` is an open string set. Never hard-code `{US, India}`. Test set adds `France` (absent from training). Every test entity, France included, must appear in the submission. | This is a deliberately planted generalization trap. |
| C | Any pretrained model used must be MIT/Apache-2.0 licensed and ≤8B parameters. | Stated model-license constraint; violating it risks disqualification on package review. |
| D | `matched_entity_ids` may only contain `S2-`/`S3-` IDs that exist in the test set. No self-matches to `S1-`. No duplicate IDs within a list. No duplicate `source1_entity_id` rows. Every test Source-1 entity must have exactly one row. | Direct validator/leaderboard rejection criteria. |

### 4. Evaluation metric

Macro-averaged **F0.5** per Source-1 entity:

```
F0.5 = (1.25 × Precision × Recall) / (0.25 × Precision + Recall)
```

Computed per Source-1 entity, then averaged across all entities. Singletons
count: an entity with no true matches scores 1.0 for an empty prediction
and 0.0 for any predicted match. Because precision is weighted 2x over
recall, false merges are costlier than missed matches — this should shape
both the blocking generosity and the final threshold choice.

**Baselines to beat, computed on your own validation split:**
- All-singletons predictor (often surprisingly strong if most Source-1
  entities truly have no match)
- Naive/random predictor

Report both alongside your model's validation F0.5 so improvement is
visible, not assumed.

### 5. Blocking / candidate generation

Blocking recall is the ceiling on everything downstream — a true match
never generated as a candidate can never be recovered by the model. Recall
should be measured on a held-out validation split **before** any modeling
work, using the ground-truth file.

**Recommended passes (union, then dedupe):**
- Normalized-name matching after suffix stripping (Pvt/Ltd/Corp/Inc/
  Private/Limited), lowercasing, punctuation removal
- **Rare name token** matching — block on tokens below a document-frequency
  threshold, not the first token of the name (first-token blocking on
  generic words like "The", "Global", "Sri", "ABC" either explodes the
  candidate set or contributes nothing; keep "first token match" as a
  *feature* instead, never a blocking key)
- Address/locality token matching (city, PIN/postal code when present)
- Country as a hard pre-filter everywhere — cross-country candidates are
  virtually never true matches, and filtering them is a safe, cheap way to
  shrink the search space without hurting recall
- Optional catch-all: TF-IDF cosine similarity (char n-grams on name +
  address) with an approximate nearest-neighbor / LSH pass to recover
  anything the token-based passes miss

**Target:** blocking recall ≥ 0.95 on validation. Below 0.90, stop and fix
blocking before touching features or the model — no amount of downstream
tuning recovers candidates that were never generated.

### 6. Pairwise features

- **Name:** Levenshtein ratio, Jaro-Winkler, token-set Jaccard, TF-IDF
  cosine (word and/or char n-gram), suffix-stripped exact-match flag,
  first-token match flag (as a feature, see above)
- **Address:** token Jaccard, city/locality match flag, PIN/postal exact-
  match flag (strong positive signal when present on both sides),
  street-number match, basic landmark-phrase handling (e.g. fuzzy matching
  against "near X" fragments)
- **Country:** exact-match flag (near-binary gate — a mismatch here should
  push a pair strongly toward negative)

### 7. Model

Start simple: Logistic Regression on the engineered pairwise features is a
fast, fully-licensable, easily-explained baseline and is often competitive
for this kind of tabular pair-classification task. Sweep the decision
threshold on the validation split and pick the value that maximizes
macro F0.5-per-entity — not accuracy, not the default 0.5. F0.5's
precision weighting typically pushes the optimal threshold higher than an
accuracy-optimal choice would. On near-ties, prefer the higher threshold.

Only after the critical path is complete and validator-passing, consider:
LightGBM/XGBoost/CatBoost for a stronger tabular fit, a small MIT/Apache-
licensed (≤8B param) sentence-embedding model for semantic name/address
similarity as an additional feature, or per-entity/per-country adaptive
thresholds.

### 8. Output format

**`matching_results.tsv`** (the only file scored on the leaderboard):

| Column | Description |
|---|---|
| `source1_entity_id` | Source 1 record ID |
| `matched_entity_ids` | Comma-separated S2-/S3- IDs, empty if none |

**`candidate_pairs.tsv`** (not scored; used to audit blocking quality):

| Column | Description |
|---|---|
| `source1_entity_id` | Source 1 record ID |
| `candidate_entity_ids` | Comma-separated candidate S2-/S3- IDs fed to the model |

Both files: one row per test Source-1 entity, tab-separated, no quoting,
comma-separated ID lists, empty string (not `None`/`NaN`) when there is
nothing to list.

### 9. Final submission package

```
<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       ├── README.md          # exact reproduction steps, data → output
│       └── requirements.txt   # pinned versions
└── Documentation_template.md  # filled-in methodology write-up
```

Methodology document must cover: methodology used, candidate generation /
blocking strategy, model architecture and feature engineering, and any
other relevant detail. No page limit — prioritize clarity and technical
depth over brevity.

### 10. Definition of done

- [ ] Recon complete, schemas and distributions reported
- [ ] Evaluator built and unit-tested against the worked example
- [ ] Baselines (all-singleton, random) computed on validation
- [ ] Blocking built, multi-pass, country-filtered
- [ ] Blocking recall measured on validation, ≥ 0.95
- [ ] Pairwise features built and unit-tested
- [ ] Model trained, threshold swept and selected on macro F0.5
- [ ] Validation F0.5 beats both baselines, reported explicitly
- [ ] `matching_results.tsv` + `candidate_pairs.tsv` generated for real test set
- [ ] Every test Source-1 ID present exactly once, empty list for singletons
- [ ] No duplicate IDs, S2-/S3- only, matches ⊆ candidates
- [ ] Official `validate_submission.py` run and returns PASS
- [ ] Model license + parameter count logged in methodology doc
- [ ] Commits made at each critical-path step, not squashed
- [ ] Final ZIP assembled per the required structure

### 11. Debugging heuristic

Whenever the model misses a true match, ask first: **was the true match
even generated as a candidate?**
- No → fix blocking (add a pass, loosen a threshold).
- Yes → the problem is downstream — check features, model, or threshold,
  in that order.

This single question resolves most debugging time faster than guessing.
