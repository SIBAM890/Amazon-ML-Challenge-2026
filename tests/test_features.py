"""Unit tests for src.entity_resolution.features (Task 1.2)."""
import os
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.entity_resolution.features import (
    FEATURES,
    build_features,
    build_features_batch,
)


@pytest.fixture(scope="module")
def tiny_vect():
    corpus = [
        "acme global services",
        "acme global services",
        "42 park street springfield",
        "zzz qqq xxx",
        "123 main road shelbyville 62704",
    ]
    return TfidfVectorizer(analyzer="char_wb",
                           ngram_range=(3, 5)).fit(corpus)


def _row(name="", addr="", country="US"):
    return {"business_name": name, "business_address": addr,
            "country": country}


def test_identical_strings_all_similarities_one(tiny_vect):
    a = _row("Acme Global Services Pvt Ltd", "42 Park Street, Springfield",
             "US")
    b = _row("Acme Global Services Pvt Ltd", "42 Park Street, Springfield",
             "US")
    f = build_features(a, b, vectorizer=tiny_vect)
    for k in ("name_levenshtein", "name_jaro_winkler", "name_token_jaccard",
              "name_char3_jaccard", "name_tfidf_cosine",
              "addr_token_jaccard", "addr_levenshtein", "addr_tfidf_cosine"):
        assert f[k] == pytest.approx(1.0), k
    assert f["name_exact"] == 1.0
    assert f["country_match"] == 1.0


def test_disjoint_strings_exact_match_features_zero(tiny_vect):
    a = _row("Acme Global Services", "42 Park Street Springfield 62704",
             "US")
    b = _row("Zzz Qqq Xxx", "7 Elm Road Shelbyville", "India")
    f = build_features(a, b, vectorizer=tiny_vect)
    assert f["name_exact"] == 0.0
    assert f["city_match"] == 0.0
    assert f["pin_match"] == 0.0
    assert f["street_num_match"] == 0.0
    assert f["country_match"] == 0.0


def test_empty_inputs_do_not_crash(tiny_vect):
    cases = [
        (_row("", "", ""), _row("", "", "")),
        (_row(None, None, None), _row(None, None, None)),
        (_row(float("nan"), float("nan"), float("nan")),
         _row("Acme", "42 Park St", "US")),
    ]
    for a, b in cases:
        f = build_features(a, b, vectorizer=tiny_vect)
        assert set(f.keys()) == set(FEATURES)
        for k, v in f.items():
            assert isinstance(v, float), k
            assert 0.0 <= v <= 1.0, (k, v)
    df = build_features_batch(
        pd.DataFrame([{"business_name": "", "business_address": "",
                       "country": ""}]),
        pd.DataFrame([{"business_name": None, "business_address": None,
                       "country": None}]),
        vectorizer=tiny_vect)
    assert df.shape == (1, 14)
    assert not df.isna().any().any()


def test_batch_output_shape_matches_input(tiny_vect):
    rows_a = [
        _row("Acme Global Services", "42 Park Street Springfield", "US"),
        _row("Foo Bar Ltd", "7 Elm Road", "India"),
        _row("", "", ""),
    ]
    rows_b = [
        _row("Acme Global Service", "42 Park St Springfield", "US"),
        _row("Baz Qux Inc", "9 Oak Ave", "France"),
        _row(None, None, None),
    ]
    df = build_features_batch(pd.DataFrame(rows_a), pd.DataFrame(rows_b),
                              vectorizer=tiny_vect)
    assert list(df.columns) == FEATURES
    assert df.shape == (3, 14)
