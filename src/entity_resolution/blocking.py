import re
import gc
import collections
from typing import List, Set
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
    suffixes = r'\b(pvt|ltd|corp|inc|private|limited|co|llc)\b'
    text = re.sub(suffixes, '', text)
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
    Builds four inverted indexes from S2 and S3 source files in two passes.

    PASS 1: frequency counting (no full-length intermediates).
    PASS 2: index construction (same chunk discipline).

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

        self._build_indexes()

    # ------------------------------------------------------------------
    # PASS 1 — frequency counting
    # ------------------------------------------------------------------

    def _count_tokens_in_chunk(self, chunk: pd.DataFrame,
                                name_ctr: collections.Counter,
                                addr_ctr: collections.Counter) -> None:
        """Update counters with per-record (deduped) tokens from this chunk."""
        for row in chunk.itertuples(index=False):
            # Name tokens
            nm = strip_legal_suffixes(normalize_text(row.business_name))
            for t in set(nm.split()):
                name_ctr[t] += 1
            # Address tokens
            for t in tokenize_address(row.business_address):
                addr_ctr[t] += 1

    # ------------------------------------------------------------------
    # PASS 2 — index construction
    # ------------------------------------------------------------------

    def _index_chunk(self, chunk: pd.DataFrame,
                     rare_name: Set[str],
                     rare_addr: Set[str]) -> None:
        """Populate all four indexes from one chunk."""
        for row in chunk.itertuples(index=False):
            eid     = row.entity_id
            country = row.country

            # --- exact name ---
            nm = strip_legal_suffixes(normalize_text(row.business_name))
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

    def get_candidates(self, s1_row: pd.Series) -> Set[str]:
        candidates: Set[str] = set()
        country = s1_row['country']

        # Pass 1: Normalized Name exact match
        norm_name = normalize_text(s1_row['business_name'])
        stripped_name = strip_legal_suffixes(norm_name)
        if stripped_name:
            candidates.update(self.exact_name_idx.get((country, stripped_name), []))

        # Pass 2: Rare name tokens
        for token in set(stripped_name.split()):
            candidates.update(self.rare_token_idx.get((country, token), []))

        # Pass 2b: Rare address tokens
        for token in tokenize_address(s1_row['business_address']):
            bucket = self.rare_addr_token_idx.get((country, token), [])
            # Defensive cap — see comment in _index_chunk above.
            if len(bucket) <= self.rare_freq_threshold:
                candidates.update(bucket)

        # Pass 3: PIN/Postal code
        for pin in extract_pin_codes(s1_row['business_address']):
            candidates.update(self.pin_idx.get((country, pin), []))

        return candidates
