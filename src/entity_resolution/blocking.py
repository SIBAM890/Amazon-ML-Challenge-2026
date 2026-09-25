import re
import gc
import collections
from typing import List, Set, Dict, Tuple
import pandas as pd

CHUNK_SIZE = 200_000


# ---------------------------------------------------------------------------
# Shared tokenizers — used identically during index-building AND query time.
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    if not isinstance(text, str) or pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    return ' '.join(text.split())


def strip_legal_suffixes(text: str) -> str:
    suffixes = (
        # Original English legal suffixes
        r'pvt|ltd|corp|inc|private|limited|co|llc|corporation|'
        # ITRANS transliterated forms of "private limited" and variants
        # produced by sanscript.transliterate(..., ITRANS) on common
        # Devanagari/Gujarati/Bengali legal endings.
        r'praiveta|praivet|limiTeDa|limiteda|limitada|limiTeda|'
        r'limitad|limita|limiteDa|'
        # Common Hindi/Devanagari legal terms (transliterated)
        # niyamita = registered/limited; samiti = society/association
        # sangh = union/association; kampani/kamapani = company
        r'niyamita|samiti|sangh|kampani|kamapani'
    )
    text = re.sub(r'\b(' + suffixes + r')\b', '', text, flags=re.IGNORECASE)
    return ' '.join(text.split())


def extract_pin_codes(text: str) -> List[str]:
    if not isinstance(text, str) or pd.isna(text):
        return []
    return re.findall(r'\b\d{5,6}\b', text)


def tokenize_address(address: str) -> Set[str]:
    if not isinstance(address, str) or pd.isna(address):
        return set()
    text = address.lower()
    text = re.sub(r'[,\.;/\-\(\)]', ' ', text)
    tokens = text.split()
    valid_tokens = set()
    for t in tokens:
        t = re.sub(r'^[^a-z0-9]+', '', t)
        t = re.sub(r'[^a-z0-9]+$', '', t)
        if len(t) >= 2 and not t.isdigit() and t.isascii():
            valid_tokens.add(t)
    return valid_tokens


def _normalize_for_tg(raw: str) -> str:
    """Full normalization pipeline for 3-gram generation.

    normalize_script -> normalize_text -> strip_legal_suffixes -> lowercase,
    then collapse spaces. Result is the string over which char-3-grams are cut.
    Import is lazy (inside function) to avoid circular dependency if
    normalization.py ever imports from blocking.py.
    """
    from src.entity_resolution.normalization import normalize_script
    t = normalize_script(str(raw) if pd.notna(raw) else "")
    t = strip_legal_suffixes(normalize_text(t))
    return t.lower().replace(" ", "")


def char_trigrams(text: str) -> Set[str]:
    """Padded character 3-grams: 'abc' -> {'_ab', 'abc', 'bc_'}."""
    norm = _normalize_for_tg(text)
    if len(norm) < 3:
        return set()
    padded = "_" + norm + "_"
    return {padded[i:i+3] for i in range(len(padded) - 2)}


def _jaccard(a: set, b: set) -> float:
    u = a | b
    return len(a & b) / len(u) if u else 0.0


# ---------------------------------------------------------------------------
# Helper: iterate a file in chunks and apply fn(chunk) -> None
# ---------------------------------------------------------------------------

def _iter_chunks(path: str, fn):
    """Read path in CHUNK_SIZE chunks, call fn(chunk) for each, then del + gc."""
    for chunk in pd.read_csv(path, sep='\t', dtype=str,
                             chunksize=CHUNK_SIZE,
                             usecols=['entity_id', 'business_name',
                                      'business_address', 'country']):
        fn(chunk)
        del chunk
        gc.collect()


# ---------------------------------------------------------------------------
# Blocker
# ---------------------------------------------------------------------------

class Blocker:
    """
    Builds five inverted indexes from S2 and S3 source files.

    Indexes:
      exact_name_idx      : (country, norm_name)  -> [entity_ids]
      rare_token_idx      : (country, name_token) -> [entity_ids]
      rare_addr_token_idx : (country, addr_token) -> [entity_ids]
      pin_idx             : (country, pin)         -> [entity_ids]
      tg_idx              : (country, trigram)     -> [entity_ids]  [Pass B]

    Pass B (3-gram) uses normalize_script (transliteration) before
    normalize_text + strip_legal_suffixes, so cross-script pairs
    (Latin vs Devanagari/Gujarati/Bengali/Tamil) produce overlapping
    3-grams.  No bucket cap on tg_idx; the rerank step (top-30 by
    Jaccard) controls candidate explosion.

    Memory constraint: never materialize >2 GB at once.  No .explode().
    No full df.copy().  del + gc.collect() after every chunk.
    """

    def __init__(self, s2_path: str, s3_path: str, rare_freq_threshold: int = 500):
        self.s2_path = s2_path
        self.s3_path = s3_path
        self.rare_freq_threshold = rare_freq_threshold

        self.exact_name_idx: dict = {}
        self.rare_token_idx: dict = {}
        self.rare_addr_token_idx: dict = {}
        self.pin_idx: dict = {}
        self.tg_idx: dict = {}          # Pass B: (country, 3gram) -> [eids]

        self._build_indexes()

    # ------------------------------------------------------------------
    # PASS 1 — frequency counting
    # ------------------------------------------------------------------

    def _count_tokens_in_chunk(self, chunk: pd.DataFrame,
                                name_ctr: collections.Counter,
                                addr_ctr: collections.Counter) -> None:
        """Update counters with per-record (deduped) tokens from this chunk."""
        from src.entity_resolution.normalization import normalize_script
        for row in chunk.itertuples(index=False):
            # Name tokens — apply transliteration before normalization
            raw_name = str(row.business_name) if pd.notna(row.business_name) else ""
            nm = strip_legal_suffixes(normalize_text(normalize_script(raw_name)))
            for t in set(nm.split()):
                name_ctr[t] += 1
            # Address tokens (non-ASCII already filtered by tokenize_address)
            for t in tokenize_address(row.business_address):
                addr_ctr[t] += 1

    # ------------------------------------------------------------------
    # PASS 2 — index construction
    # ------------------------------------------------------------------

    def _index_chunk(self, chunk: pd.DataFrame,
                     rare_name: Set[str],
                     rare_addr: Set[str]) -> None:
        """Populate all five indexes from one chunk."""
        from src.entity_resolution.normalization import normalize_script
        for row in chunk.itertuples(index=False):
            eid     = row.entity_id
            country = row.country
            raw_name = str(row.business_name) if pd.notna(row.business_name) else ""

            # --- exact name (with transliteration) ---
            nm = strip_legal_suffixes(normalize_text(normalize_script(raw_name)))
            if nm:
                key = (country, nm)
                if key in self.exact_name_idx:
                    self.exact_name_idx[key].append(eid)
                else:
                    self.exact_name_idx[key] = [eid]

            # --- rare name tokens ---
            for t in set(nm.split()):
                if t in rare_name:
                    key = (country, t)
                    if key in self.rare_token_idx:
                        self.rare_token_idx[key].append(eid)
                    else:
                        self.rare_token_idx[key] = [eid]

            # --- rare address tokens ---
            for t in tokenize_address(row.business_address):
                if t in rare_addr:
                    key = (country, t)
                    if key in self.rare_addr_token_idx:
                        bucket = self.rare_addr_token_idx[key]
                        # Defensive cap: frequency threshold already guarantees
                        # <= rare_freq_threshold records per token, so this
                        # cannot normally fire at current settings.
                        if len(bucket) < self.rare_freq_threshold:
                            bucket.append(eid)
                    else:
                        self.rare_addr_token_idx[key] = [eid]

            # --- PIN codes ---
            for p in extract_pin_codes(row.business_address):
                key = (country, p)
                if key in self.pin_idx:
                    self.pin_idx[key].append(eid)
                else:
                    self.pin_idx[key] = [eid]

            # --- Pass B: character 3-gram index (no bucket cap) ---
            for tg in char_trigrams(raw_name):
                key = (country, tg)
                if key in self.tg_idx:
                    self.tg_idx[key].append(eid)
                else:
                    self.tg_idx[key] = [eid]

    # ------------------------------------------------------------------
    # Orchestrator
    # ------------------------------------------------------------------

    def _build_indexes(self) -> None:
        # ---- PASS 1: frequency ----------------------------------------
        name_ctr: collections.Counter = collections.Counter()
        addr_ctr: collections.Counter = collections.Counter()

        def _count(chunk):
            self._count_tokens_in_chunk(chunk, name_ctr, addr_ctr)

        _iter_chunks(self.s2_path, _count)
        _iter_chunks(self.s3_path, _count)

        rare_name = {t for t, c in name_ctr.items() if c <= self.rare_freq_threshold}
        rare_addr = {t for t, c in addr_ctr.items() if c <= self.rare_freq_threshold}
        del name_ctr, addr_ctr
        gc.collect()

        # ---- PASS 2: index construction --------------------------------
        def _index(chunk):
            self._index_chunk(chunk, rare_name, rare_addr)

        _iter_chunks(self.s2_path, _index)
        _iter_chunks(self.s3_path, _index)

        del rare_name, rare_addr
        gc.collect()

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get_candidates(self, s1_row: pd.Series) -> Tuple[Set[str], Dict[str, str]]:
        """Return (final_candidate_set, {eid: pass_name}) for one S1 record.

        Rerank rule:
          final = high_confidence_candidates   (exact name + PIN — unlimited)
                  ∪ top-30 Pass-B candidates by Jaccard >= 0.2 (descending)
        """
        from src.entity_resolution.normalization import normalize_script
        country = s1_row['country']
        raw_name = str(s1_row['business_name']) if pd.notna(s1_row['business_name']) else ""

        # Normalize name with transliteration
        norm_name    = normalize_text(normalize_script(raw_name))
        stripped_name = strip_legal_suffixes(norm_name)

        pass_source: Dict[str, str] = {}   # eid -> first pass that found it

        # ----------------------------------------------------------------
        # High-confidence passes (guaranteed slots, bypass top-30 cut)
        # ----------------------------------------------------------------

        # Pass 1: Normalized Name exact match
        if stripped_name:
            for eid in self.exact_name_idx.get((country, stripped_name), []):
                pass_source.setdefault(eid, 'exact_name')

        # Pass 3: PIN/Postal code
        for pin in extract_pin_codes(s1_row['business_address']):
            for eid in self.pin_idx.get((country, pin), []):
                pass_source.setdefault(eid, 'pin')

        high_conf_set: Set[str] = set(pass_source.keys())

        # ----------------------------------------------------------------
        # Low-precision passes (contribute to pool, subject to rerank cut)
        # ----------------------------------------------------------------

        # Pass 2: Rare name tokens
        for token in set(stripped_name.split()):
            for eid in self.rare_token_idx.get((country, token), []):
                pass_source.setdefault(eid, 'rare_token')

        # Pass 2b: Rare address tokens
        for token in tokenize_address(s1_row['business_address']):
            bucket = self.rare_addr_token_idx.get((country, token), [])
            # Defensive cap — see comment in _index_chunk above.
            if len(bucket) <= self.rare_freq_threshold:
                for eid in bucket:
                    pass_source.setdefault(eid, 'rare_addr_token')

        # Pass B: Character 3-gram (Jaccard >= 0.2, top-30 kept)
        s1_tgs = char_trigrams(raw_name)
        tg_cand_counts: Dict[str, int] = {}   # eid -> shared 3-gram count
        tg_cand_union:  Dict[str, int] = {}   # eid -> union 3-gram count (denominator)
        if s1_tgs:
            for tg in s1_tgs:
                for eid in self.tg_idx.get((country, tg), []):
                    tg_cand_counts[eid] = tg_cand_counts.get(eid, 0) + 1

            # Compute Jaccard for each candidate that shared >= 1 3-gram
            # Jaccard = shared / (|s1_tgs| + |cand_tgs| - shared)
            # We don't have cand_tgs stored, so approximate:
            #   Jaccard_approx = shared / (|s1_tgs| + shared_as_proxy)
            # For precision: we do full Jaccard only for candidates above
            # a cheap pre-filter of shared >= 2, to avoid calling
            # char_trigrams() for every single candidate.
            s1_tg_size = len(s1_tgs)
            pass_b_scored = []
            for eid, shared in tg_cand_counts.items():
                if shared < 2:
                    continue   # cheap pre-filter: skip 1-gram overlaps
                # Jaccard lower-bound: shared / (s1_tg_size + shared)
                # is always <= true Jaccard, so use it as a fast guard.
                jac_lb = shared / (s1_tg_size + shared)
                if jac_lb >= 0.2:
                    pass_b_scored.append((eid, jac_lb))

            # Sort descending by Jaccard, keep top 30
            pass_b_scored.sort(key=lambda x: -x[1])
            pass_b_top30 = {eid for eid, _ in pass_b_scored[:30]}

            for eid in pass_b_top30:
                pass_source.setdefault(eid, 'pass_b')
        else:
            pass_b_top30 = set()

        # ----------------------------------------------------------------
        # Rerank: UNION of high-confidence (unlimited) + Pass-B top-30
        # ----------------------------------------------------------------
        final = high_conf_set | pass_b_top30

        return final, pass_source
