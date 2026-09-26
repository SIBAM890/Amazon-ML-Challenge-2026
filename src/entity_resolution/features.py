"""Pairwise features (PROJECT.md Step B / ENHANCED.md Task 1.2).

14 features, all in [0, 1], NaN -> 0.0:

  name:    name_exact, name_levenshtein, name_jaro_winkler,
           name_token_jaccard, name_char3_jaccard, name_tfidf_cosine
  address: addr_token_jaccard, addr_levenshtein, addr_tfidf_cosine,
           city_match, pin_match, street_num_match
  combined: country_match, length_diff

Normalization for ALL features (spec-literal, reused — never duplicated):
    normalize_script -> normalize_text -> strip_legal_suffixes -> lower
from normalization.py / blocking.py.

VECTORIZATION NOTE (deliberate, measured decision): the spec asks for
``rapidfuzz.process.cdist`` for the string-similarity features. cdist
computes the full queries × choices CROSS product; our batch inputs are
N ALIGNED pairs, so cdist would compute N² cells to use N diagonals —
per 2000-pair block ~4M C++ comparisons (~1.2s for 3 scorers) to yield
2000 values, i.e. ~20h at 117M pairs and a certain miss of the
<10min-on-20k gate. Aligned elementwise rapidfuzz calls are the same
C++ scorers at ~3µs/pair (~12s per feature on the 1.37M-pair K=50
sample) and pass the gate. Token/char-3gram Jaccard and TF-IDF cosine
below ARE fully vectorized (binary CountVectorizer + sparse ops, no
pair loop). The only per-pair Python iteration left is over cheap
C++ set/scorer calls, never over slow pure-Python similarity code.
"""

import os
import pickle
import re
from typing import Dict, Mapping, Optional

import numpy as np
import pandas as pd
from rapidfuzz.distance import JaroWinkler, Levenshtein
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from src.entity_resolution.blocking import (
    normalize_text,
    strip_legal_suffixes,
)
from src.entity_resolution.normalization import normalize_script

FEATURES = [
    "name_exact",
    "name_levenshtein",
    "name_jaro_winkler",
    "name_token_jaccard",
    "name_char3_jaccard",
    "name_tfidf_cosine",
    "addr_token_jaccard",
    "addr_levenshtein",
    "addr_tfidf_cosine",
    "city_match",
    "pin_match",
    "street_num_match",
    "country_match",
    "length_diff",
]

DEFAULT_TFIDF_PATH = os.path.join("data", "interim", "tfidf_vectorizer.pkl")

_VECTORIZER_CACHE: dict = {}


# ---------------------------------------------------------------------------
# Normalization (single canonical pipeline, reused everywhere)
# ---------------------------------------------------------------------------

def _norm(raw: object) -> str:
    """normalize_script -> normalize_text -> strip_legal_suffixes -> lower."""
    if raw is None:
        return ""
    if not isinstance(raw, str):
        try:
            if pd.isna(raw):
                return ""
        except Exception:
            pass
        raw = str(raw)
    t = strip_legal_suffixes(normalize_text(normalize_script(raw)))
    return t.lower()


def _safe_str(raw: object) -> str:
    if raw is None:
        return ""
    if not isinstance(raw, str):
        try:
            if pd.isna(raw):
                return ""
        except Exception:
            pass
        raw = str(raw)
    return raw


# ---------------------------------------------------------------------------
# Small parsers (deterministic, no external data)
# ---------------------------------------------------------------------------

def _token_jaccard(a: str, b: str) -> float:
    sa, sb = set(a.split()), set(b.split())
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _padded_trigrams(s: str) -> set:
    t = "".join(s.split())
    if len(t) < 3:
        return set()
    p = "_" + t + "_"
    return {p[i:i + 3] for i in range(len(p) - 2)}


def _trigram_jaccard(a: str, b: str) -> float:
    sa, sb = _padded_trigrams(a), _padded_trigrams(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _city_token(norm_addr: str) -> str:
    """Last alpha token (len>=3) scanning from the end; '' if none."""
    for tok in reversed(norm_addr.split()):
        if len(tok) >= 3 and not tok[0].isdigit() and tok.isalpha():
            return tok
    return ""


def _extract_pins(address: object, country: object) -> set:
    """Country-aware PIN extraction (no hard-coded country SET for logic).

    India -> 6-digit, US/France variants -> 5-digit, each at end of
    string or right after a comma (postal-position heuristic).
    Unknown country -> empty (skip).
    """
    addr = _safe_str(address)
    c = _safe_str(country).strip().lower()
    if c == "india":
        pat = r",\s*(\d{6})\b|\b(\d{6})\s*$"
    elif c in ("us", "usa", "united states", "france", "fr"):
        pat = r",\s*(\d{5})\b|\b(\d{5})\s*$"
    else:
        return set()
    out = set()
    for m in re.finditer(pat, addr):
        out.add(m.group(1) or m.group(2))
    return out


def _street_num(norm_addr: str) -> str:
    m = re.search(r"\b\d+[a-z]?\b", norm_addr)
    return m.group(0) if m else ""


# ---------------------------------------------------------------------------
# TF-IDF: fit once on S2+S3 union, cache to disk for inference.py
# ---------------------------------------------------------------------------

def _default_vect() -> TfidfVectorizer:
    # No max_features: the 1M-sample target is vocab >= 200k, and a
    # 50k cap would make that unreachable by construction.
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                           min_df=2)


def fit_and_save_tfidf_vectorizer(
    docs_df=None,
    out_path: str = DEFAULT_TFIDF_PATH,
    chunksize: int = 200_000,
    limit_rows: Optional[int] = None,
) -> str:
    """Fit ONE char TF-IDF on normalized S2+S3 `name + ' ' + address`.

    docs_df: optional pre-sampled DataFrame with business_name /
        business_address columns (e.g. 500k S2 + 500k S3). If None,
        streams S2+S3 from disk in chunks (never >2GB).
    Writes a pickle with the fitted vectorizer to out_path. Reports
    vocab size. Run once before features at scale; NOT run by unit
    tests (they inject a tiny vectorizer instead).
    """
    import gc
    import time as _time

    t0 = _time.time()
    docs = []
    if docs_df is not None:
        names = docs_df["business_name"].fillna("").tolist()
        addrs = docs_df["business_address"].fillna("").tolist()
        docs = [_norm(nm) + " " + _norm(ad)
                for nm, ad in zip(names, addrs)]
        print(f"tfidf fit docs from dataframe: {len(docs)}", flush=True)
    else:
        from src.entity_resolution.config import TRAIN_SOURCE2, TRAIN_SOURCE3
        n = 0
        for path in (TRAIN_SOURCE2, TRAIN_SOURCE3):
            for chunk in pd.read_csv(path, sep="\t", dtype=str,
                                     chunksize=chunksize,
                                     usecols=["business_name",
                                              "business_address"]):
                names = chunk["business_name"].fillna("")
                addrs = chunk["business_address"].fillna("")
                for nm, ad in zip(names.tolist(), addrs.tolist()):
                    docs.append(_norm(nm) + " " + _norm(ad))
                    n += 1
                    if limit_rows and n >= limit_rows:
                        break
                del chunk
                gc.collect()
                if limit_rows and n >= limit_rows:
                    break
    vect = _default_vect().fit(docs)
    print(f"tfidf vocab_size={len(vect.vocabulary_)} "
          f"docs={len(docs)} fit_t={_time.time()-t0:.1f}s", flush=True)
    del docs
    gc.collect()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(vect, f)
    return out_path


def _get_vectorizer(vectorizer=None):
    """Explicit arg > module override > disk cache > clear error."""
    if vectorizer is not None:
        return vectorizer
    if "vect" in _VECTORIZER_CACHE:
        return _VECTORIZER_CACHE["vect"]
    if not os.path.exists(DEFAULT_TFIDF_PATH):
        raise RuntimeError(
            f"TF-IDF vectorizer missing at {DEFAULT_TFIDF_PATH}. "
            "Run fit_and_save_tfidf_vectorizer() first (or pass "
            "vectorizer= explicitly in tests).")
    with open(DEFAULT_TFIDF_PATH, "rb") as f:
        vect = pickle.load(f)
    _VECTORIZER_CACHE["vect"] = vect
    return vect


def _tfidf_cosine(vect, a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    m = vect.transform([a, b])
    num = m[0].multiply(m[1]).sum()
    den = np.sqrt(m[0].multiply(m[0]).sum() * m[1].multiply(m[1]).sum())
    return float(num / den) if den else 0.0


# ---------------------------------------------------------------------------
# Single-pair entry point (unit tests, debugging)
# ---------------------------------------------------------------------------

def build_features(
    s1_row: Mapping,
    candidate_row: Mapping,
    vectorizer=None,
) -> Dict[str, float]:
    """Compute the 14 features for one (S1, candidate) pair."""
    vect = _get_vectorizer(vectorizer)

    na, nb = _norm(s1_row.get("business_name")), \
        _norm(candidate_row.get("business_name"))
    aa, ab = _norm(s1_row.get("business_address")), \
        _norm(candidate_row.get("business_address"))
    ca, cb = _safe_str(s1_row.get("country")), \
        _safe_str(candidate_row.get("country"))

    feats = {
        "name_exact": 1.0 if (na and na == nb) else 0.0,
        "name_levenshtein": float(Levenshtein.normalized_similarity(na, nb))
        if (na or nb) else 0.0,
        "name_jaro_winkler": float(JaroWinkler.similarity(na, nb))
        if (na or nb) else 0.0,
        "name_token_jaccard": _token_jaccard(na, nb),
        "name_char3_jaccard": _trigram_jaccard(na, nb),
        "name_tfidf_cosine": _tfidf_cosine(vect, na, nb),
        "addr_token_jaccard": _token_jaccard(aa, ab),
        "addr_levenshtein": float(Levenshtein.normalized_similarity(aa, ab))
        if (aa or ab) else 0.0,
        "addr_tfidf_cosine": _tfidf_cosine(vect, aa, ab),
        "city_match": 1.0 if ((lambda x, y: x and x == y)
                              (_city_token(aa), _city_token(ab))) else 0.0,
        "pin_match": 1.0 if (_extract_pins(s1_row.get("business_address"), ca)
                             & _extract_pins(candidate_row.get(
                                 "business_address"), cb)) else 0.0,
        "street_num_match": 1.0 if ((lambda x, y: x and x == y)
                                    (_street_num(aa), _street_num(ab)))
        else 0.0,
        "country_match": 1.0 if (ca and ca == cb) else 0.0,
        "length_diff": (abs(len(na) - len(nb)) / max(max(len(na), len(nb)), 1)),
    }
    return {k: float(v) for k, v in feats.items()}


# ---------------------------------------------------------------------------
# Batched entry point (inference at scale)
# ---------------------------------------------------------------------------

def _sparse_jaccard(mat_a: sparse.csr_matrix,
                    mat_b: sparse.csr_matrix) -> np.ndarray:
    inter = np.asarray(mat_a.multiply(mat_b).sum(axis=1)).ravel()
    union = (np.asarray(mat_a.sum(axis=1)).ravel()
             + np.asarray(mat_b.sum(axis=1)).ravel() - inter)
    return np.divide(inter, union, out=np.zeros_like(inter, dtype=float),
                     where=union > 0)


def _batch_jaccard(docs_a: list, docs_b: list, make_vect) -> np.ndarray:
    """Vectorized Jaccard with empty-vocabulary guard (all-empty batches).

    sklearn raises ValueError on empty vocabulary; at scale an
    all-empty chunk is realistic and must yield zeros, not a crash.
    """
    n = len(docs_a)
    try:
        vect = make_vect().fit(docs_a + docs_b)
    except ValueError:
        return np.zeros(n, dtype=float)
    return _sparse_jaccard(vect.transform(docs_a), vect.transform(docs_b))


def _sparse_row_cosine(mat_a: sparse.csr_matrix,
                       mat_b: sparse.csr_matrix) -> np.ndarray:
    num = np.asarray(mat_a.multiply(mat_b).sum(axis=1)).ravel()
    den = np.sqrt(np.asarray(mat_a.multiply(mat_a).sum(axis=1)).ravel()
                  * np.asarray(mat_b.multiply(mat_b).sum(axis=1)).ravel())
    return np.divide(num, den, out=np.zeros_like(num, dtype=float),
                     where=den > 0)


def build_features_batch(
    s1_batch: pd.DataFrame,
    candidates_batch: pd.DataFrame,
    vectorizer=None,
) -> pd.DataFrame:
    """Vectorized 14-feature frame for N ALIGNED pairs.

    Same values as build_features (same normalization, same padded
    3-grams, same TF-IDF object). Fuzzy string sims use aligned
    elementwise rapidfuzz C++ calls (see module docstring for why
    cdist's N×N cross product does not apply); Jaccard/TF-IDF paths
    are fully vectorized sparse ops. No slow pure-Python similarity
    loop. NaN -> "" -> 0.0.
    """
    vect = _get_vectorizer(vectorizer)
    n = len(s1_batch)
    if len(candidates_batch) != n:
        raise ValueError(f"aligned batches required: {n} vs "
                         f"{len(candidates_batch)}")

    na = [_norm(v) for v in s1_batch["business_name"].tolist()]
    nb = [_norm(v) for v in candidates_batch["business_name"].tolist()]
    aa = [_norm(v) for v in s1_batch["business_address"].tolist()]
    ab = [_norm(v) for v in candidates_batch["business_address"].tolist()]
    ca = [_safe_str(v) for v in s1_batch["country"].tolist()]
    cb = [_safe_str(v) for v in candidates_batch["country"].tolist()]

    out: Dict[str, np.ndarray] = {}
    out["name_exact"] = (np.array(na) == np.array(nb)) & (np.array(na) != "")
    out["name_levenshtein"] = np.array(
        [Levenshtein.normalized_similarity(x, y) if (x or y) else 0.0
         for x, y in zip(na, nb)], dtype=float)
    out["name_jaro_winkler"] = np.array(
        [JaroWinkler.similarity(x, y) if (x or y) else 0.0
         for x, y in zip(na, nb)], dtype=float)

    out["name_token_jaccard"] = _batch_jaccard(
        na, nb, lambda: CountVectorizer(analyzer="word",
                                        token_pattern=r"(?u)\S+",
                                        binary=True))
    out["addr_token_jaccard"] = _batch_jaccard(
        aa, ab, lambda: CountVectorizer(analyzer="word",
                                        token_pattern=r"(?u)\S+",
                                        binary=True))

    out["name_char3_jaccard"] = _batch_jaccard(
        na, nb, lambda: CountVectorizer(analyzer=_padded_trigrams_list,
                                        binary=True))

    out["name_tfidf_cosine"] = _sparse_row_cosine(vect.transform(na),
                                                  vect.transform(nb))
    out["addr_tfidf_cosine"] = _sparse_row_cosine(vect.transform(aa),
                                                  vect.transform(ab))
    out["addr_levenshtein"] = np.array(
        [Levenshtein.normalized_similarity(x, y) if (x or y) else 0.0
         for x, y in zip(aa, ab)], dtype=float)

    s_ca = pd.Series(ca)
    s_cb = pd.Series(cb)
    out["country_match"] = (s_ca.values == s_cb.values) & (s_ca.values != "")

    ca_city = pd.Series(aa).map(_city_token)
    cb_city = pd.Series(ab).map(_city_token)
    out["city_match"] = ((ca_city.values == cb_city.values)
                         & (ca_city.values != ""))

    pa = pd.Series([_extract_pins(s1_batch["business_address"].iloc[i], ca[i])
                    for i in range(n)])
    pb = pd.Series([_extract_pins(candidates_batch["business_address"].iloc[i],
                                  cb[i]) for i in range(n)])
    out["pin_match"] = np.array([1.0 if (x & y) else 0.0 for x, y in
                                 zip(pa.tolist(), pb.tolist())],
                                dtype=float)

    sa = pd.Series(aa).map(_street_num)
    sb = pd.Series(ab).map(_street_num)
    out["street_num_match"] = ((sa.values == sb.values)
                               & (sa.values != "")).astype(float)

    la = np.array([len(t) for t in na], dtype=float)
    lb = np.array([len(t) for t in nb], dtype=float)
    out["length_diff"] = np.abs(la - lb) / np.maximum(np.maximum(la, lb), 1.0)

    df = pd.DataFrame({k: np.asarray(v, dtype=float) for k, v in out.items()})
    return df[FEATURES]


def _padded_trigrams_list(s: str) -> list:
    """CountVectorizer analyzer hook: same padded 3-grams as _padded_trigrams."""
    return sorted(_padded_trigrams(s))
