"""Retrieval stage (PROJECT.md Step A / ENHANCED.md Task 1.1).

Cuts the blocking candidate pool (~210/S1) down toward ~50/S1 before
feature computation, using the pure-union pattern (lesson from the
rerank bug that collapsed recall 0.79 -> 0.44):

    final = (all exact_name candidates)
          ∪ (all PIN candidates)
          ∪ (top-K of the remaining pool by cheap similarity)

Cheap similarity (PROJECT.md Step A):

    cheap_similarity = jaccard_3gram(name + " " + address)
                     + 0.2 * country_match

The 3-gram Jaccard here is deliberately cheap: lowercase,
non-alphanumeric folding, padded 3-grams, NO transliteration and NO
legal-suffix stripping. Cross-script/near-duplicate handling already
happened in blocking (exact pass with ITRANS); retrieval only ranks.

Guaranteed slots (exact_name, PIN) use the SAME normalization as
blocking.py (normalize_script -> normalize_text ->
strip_legal_suffixes; PIN via extract_pin_codes), so retrieval never
drops what blocking guaranteed.

Does NOT modify blocking.py (imports helpers only).
"""

import re
from typing import Dict, Iterable, Mapping, Set, Tuple

from src.entity_resolution.blocking import (
    extract_pin_codes,
    normalize_text,
    strip_legal_suffixes,
)
from src.entity_resolution.normalization import normalize_script

COUNTRY_BONUS = 0.2


def _norm_name(raw: object) -> str:
    """Blocking-identical normalized name for exact-match comparison."""
    if raw is None:
        return ""
    try:
        import pandas as pd  # local import: keep module pandas-free
        if pd.isna(raw):
            return ""
    except Exception:
        pass
    if not isinstance(raw, str):
        raw = str(raw)
    return strip_legal_suffixes(normalize_text(normalize_script(raw)))


def _pins(raw: object) -> Set[str]:
    """PIN set for an address (blocking-identical extractor)."""
    if raw is None:
        return set()
    if not isinstance(raw, str):
        try:
            import pandas as pd
            if pd.isna(raw):
                return set()
        except Exception:
            pass
        raw = str(raw)
    return set(extract_pin_codes(raw))


def _cheap_trigrams(text: object) -> Set[str]:
    """Padded char 3-grams over cheap normalization (no transliteration).

    lower -> non [a-z0-9] becomes space -> collapse spaces ->
    remove spaces -> "_" + s + "_" -> 3-grams. Empty/short -> set().
    """
    if text is None:
        return set()
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    text = " ".join(text.split()).replace(" ", "")
    if len(text) < 3:
        return set()
    padded = "_" + text + "_"
    return {padded[i:i + 3] for i in range(len(padded) - 2)}


def _jaccard(a: Set[str], b: Set[str]) -> float:
    u = len(a | b)
    return len(a & b) / u if u else 0.0


def _as_str(v: object) -> str:
    return "" if v is None else (v if isinstance(v, str) else str(v))


def retrieve(
    s1_row: Mapping,
    blocking_candidates: Iterable[str],
    s2s3_lookup: Mapping[str, Mapping],
    K: int = 50,
) -> Tuple[Set[str], Dict]:
    """Rank blocking candidates down to guaranteed ∪ top-K.

    Args:
        s1_row: mapping with business_name, business_address, country
            (and optionally entity_id, used only defensively to drop a
            self-match).
        blocking_candidates: candidate S2/S3 ids from blocking.
        s2s3_lookup: candidate id -> mapping with business_name,
            business_address, country.
        K: how many of the non-guaranteed pool to keep by cheap score.

    Returns:
        (final_candidates, metadata) where metadata holds K, pool sizes
        and the guaranteed/ranked split for diagnostics.
    """
    # -- normalize S1 side (blocking-identical for exact/PIN) -----------
    s1_name = _as_str(s1_row.get("business_name", ""))
    s1_addr = _as_str(s1_row.get("business_address", ""))
    s1_country = _as_str(s1_row.get("country", ""))
    s1_id = s1_row.get("entity_id", None)
    s1_norm = _norm_name(s1_name)
    s1_pins = _pins(s1_addr)
    s1_tris = _cheap_trigrams(s1_name + " " + s1_addr)

    # -- dedupe blocking pool --------------------------------------------
    pool: Set[str] = set()
    for c in blocking_candidates or []:
        if c is None:
            continue
        c = c if isinstance(c, str) else str(c)
        if not c or c == "nan":
            continue
        if s1_id is not None and c == s1_id:
            continue
        pool.add(c)

    # -- guaranteed slots: exact normalized name OR PIN overlap ----------
    guaranteed: Set[str] = set()
    n_exact = 0
    n_pin = 0
    for cid in pool:
        rec = s2s3_lookup.get(cid)
        if rec is None:
            continue
        c_country = _as_str(rec.get("country", ""))
        if s1_norm and _norm_name(rec.get("business_name", "")) == s1_norm \
                and (not s1_country or not c_country or c_country == s1_country):
            guaranteed.add(cid)
            n_exact += 1
            continue
        if s1_pins and (s1_pins & _pins(rec.get("business_address", ""))):
            guaranteed.add(cid)
            n_pin += 1

    # -- rank the rest by cheap score ------------------------------------
    # Candidates missing from the lookup cannot be scored; keep them
    # (union principle: never drop what you cannot judge).
    remaining = pool - guaranteed
    scored = []
    n_unscored = 0
    for cid in remaining:
        rec = s2s3_lookup.get(cid)
        if rec is None:
            n_unscored += 1
            continue
        c_country = _as_str(rec.get("country", ""))
        c_tris = _cheap_trigrams(
            _as_str(rec.get("business_name", ""))
            + " "
            + _as_str(rec.get("business_address", ""))
        )
        score = _jaccard(s1_tris, c_tris)
        if s1_country and c_country and s1_country == c_country:
            score += COUNTRY_BONUS
        scored.append((score, cid))
    scored.sort(key=lambda t: (-t[0], t[1]))

    k = max(0, int(K))
    topk = {cid for _, cid in scored[:k]} if k else set()
    unscored = {cid for cid in remaining if s2s3_lookup.get(cid) is None}

    final = guaranteed | topk | unscored
    metadata = {
        "K": k,
        "n_blocking": len(pool),
        "n_exact": n_exact,
        "n_pin": n_pin,
        "n_guaranteed": len(guaranteed),
        "n_ranked": len(topk),
        "n_unscored": len(unscored),
        "n_final": len(final),
    }
    return final, metadata
