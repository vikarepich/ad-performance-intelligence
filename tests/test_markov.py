"""
Unit tests for Markov chain attribution.

Test strategy
-------------
- Toy data with hand-computed answers — covers the math.
- Real data with empirical invariants — covers integration.
- Edge cases (missing columns, unknown channels) — covers robustness.

Each test is self-contained and uses fixtures for shared data.
Run from project root:  pytest tests/test_markov.py -v
"""
import numpy as np
import pandas as pd
import pytest

from src.attribution.markov_model import (
    build_paths,
    build_transition_matrix,
    conversion_probability,
    removal_effect,
    markov_attribution,
    START,
    CONVERSION,
    NULL,
)


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture
def toy_journeys() -> pd.DataFrame:
    """3-user toy DataFrame with hand-computed expected outputs.

    User A: tiktok -> meta -> google -> CONVERTS
    User B: meta -> google -> NULL
    User C: tiktok -> CONVERTS
    """
    return pd.DataFrame([
        {"user_id": "A", "touch_index": 0, "channel": "tiktok",  "is_converting_touch": False, "revenue": 0},
        {"user_id": "A", "touch_index": 1, "channel": "meta",    "is_converting_touch": False, "revenue": 0},
        {"user_id": "A", "touch_index": 2, "channel": "google",  "is_converting_touch": True,  "revenue": 100.0},
        {"user_id": "B", "touch_index": 0, "channel": "meta",    "is_converting_touch": False, "revenue": 0},
        {"user_id": "B", "touch_index": 1, "channel": "google",  "is_converting_touch": False, "revenue": 0},
        {"user_id": "C", "touch_index": 0, "channel": "tiktok",  "is_converting_touch": True,  "revenue": 50.0},
    ])


@pytest.fixture
def toy_paths(toy_journeys):
    return build_paths(toy_journeys)


@pytest.fixture
def toy_matrix(toy_paths):
    return build_transition_matrix(toy_paths)


@pytest.fixture(scope="module")
def real_journeys() -> pd.DataFrame:
    """The 2000-user simulated journey dataset shipped with the project."""
    return pd.read_csv("data/processed/user_journeys.csv")


# ============================================================
# build_paths
# ============================================================

def test_build_paths_toy(toy_paths):
    """Paths must match the hand-computed expected sequence."""
    expected = [
        [START, "tiktok", "meta", "google", CONVERSION],
        [START, "meta", "google", NULL],
        [START, "tiktok", CONVERSION],
    ]
    assert toy_paths == expected


def test_build_paths_order_independence(toy_journeys):
    """Shuffling input rows must not change resulting paths.

    This guards against a real production bug: if upstream merges or
    parquet reads ever return rows out of touch_index order, our
    sort_values inside build_paths must put them back.
    """
    expected = build_paths(toy_journeys)
    shuffled = toy_journeys.sample(frac=1, random_state=42).reset_index(drop=True)
    assert build_paths(shuffled) == expected


def test_build_paths_missing_columns_raises(toy_journeys):
    """Missing required column must raise ValueError, not silently
    return wrong results."""
    bad = toy_journeys.drop(columns=["touch_index"])
    with pytest.raises(ValueError, match="missing columns"):
        build_paths(bad)


# ============================================================
# build_transition_matrix
# ============================================================

def test_transition_matrix_rows_sum_to_one(toy_matrix):
    """Every row must sum to 1.0 (probability conservation).

    For absorbing states, the self-loop construction ensures this
    even though they have no outgoing transitions in the data.
    """
    assert np.allclose(toy_matrix.sum(axis=1).values, 1.0)


def test_transition_matrix_cells_match_handcomputed(toy_matrix):
    """Specific transition probabilities must match the hand-computed values.

    From toy paths:
      START emits 3 transitions (2 -> tiktok, 1 -> meta)
      tiktok emits 2 (1 -> meta, 1 -> CONVERSION)
      meta emits 2 (both -> google)
      google emits 2 (1 -> CONVERSION, 1 -> NULL)
    """
    expected = {
        (START, "tiktok"):       2 / 3,
        (START, "meta"):         1 / 3,
        ("tiktok", "meta"):      0.5,
        ("tiktok", CONVERSION):  0.5,
        ("meta", "google"):      1.0,
        ("google", CONVERSION):  0.5,
        ("google", NULL):        0.5,
        # Absorbing self-loops
        (CONVERSION, CONVERSION): 1.0,
        (NULL, NULL):             1.0,
    }
    for (frm, to), want in expected.items():
        got = toy_matrix.loc[frm, to]
        assert np.isclose(got, want), f"M[{frm!r}, {to!r}] = {got}, expected {want}"


# ============================================================
# conversion_probability
# ============================================================

def test_conversion_probability_toy_handcomputed(toy_matrix):
    """P(reach CONVERSION from START) on toy paths.

    By hand: 2 of 3 toy users converted, so empirical rate = 2/3.
    Markov derived from same data must match exactly.
    """
    p = conversion_probability(toy_matrix)
    assert np.isclose(p, 2 / 3)


def test_conversion_probability_matches_empirical_on_real_data(real_journeys):
    """Critical invariant: Markov-derived conversion probability must equal
    the empirical conversion rate on the same data.

    Both quantities come from the same observations, so any divergence
    is a bug in build_transition_matrix or conversion_probability.
    """
    paths = build_paths(real_journeys)
    M = build_transition_matrix(paths)
    p_markov = conversion_probability(M)

    n_users = real_journeys["user_id"].nunique()
    n_converters = real_journeys.loc[
        real_journeys["is_converting_touch"] == True, "user_id"
    ].nunique()
    p_empirical = n_converters / n_users

    assert abs(p_markov - p_empirical) < 1e-3, (
        f"Markov conversion prob {p_markov:.6f} does not match "
        f"empirical rate {p_empirical:.6f}"
    )


# ============================================================
# removal_effect
# ============================================================

def test_removal_effect_handles_unknown_channel(toy_matrix):
    """Asking about a channel that was never seen must not crash.
    Returns 0.0 (no effect — channel didn't exist in data anyway).
    """
    assert removal_effect(toy_matrix, "channel_that_never_existed") == 0.0


def test_removal_effect_in_zero_one_range(real_journeys):
    """On real data, every channel's removal effect must be in [0, 1].

    Negative values would mean the channel hurts conversion (impossible
    by construction since we redirect to NULL). Values > 1 would mean
    removing one channel kills more than 100% of conversions (also
    impossible).
    """
    paths = build_paths(real_journeys)
    M = build_transition_matrix(paths)
    channels = [s for s in M.index if s not in (START, CONVERSION, NULL)]

    for ch in channels:
        re = removal_effect(M, ch)
        assert 0.0 <= re <= 1.0 + 1e-9, f"removal_effect({ch}) = {re} out of [0, 1]"


# ============================================================
# markov_attribution end-to-end
# ============================================================

def test_markov_attribution_revenue_conservation(real_journeys):
    """Total attributed revenue must equal total converting revenue.

    This mirrors the same invariant we check for last_click / linear /
    time_decay / shapley — it's the basic 'no money lost or invented'
    check that distinguishes a working attribution model from a broken one.
    """
    result = markov_attribution(real_journeys)
    total_revenue = real_journeys.loc[
        real_journeys["is_converting_touch"] == True, "revenue"
    ].sum()
    assert np.isclose(result["attributed_revenue"].sum(), total_revenue)


def test_markov_attribution_share_sums_to_one(real_journeys):
    """Channel shares must sum to 1.0 — no double-counting, no loss."""
    result = markov_attribution(real_journeys)
    assert np.isclose(result["share"].sum(), 1.0)


def test_markov_attribution_schema_compatible_with_other_models(real_journeys):
    """Markov output must include the same columns as last_click / shapley,
    so the analyzer can merge results into one comparison table."""
    from src.attribution.models import last_click

    mk = markov_attribution(real_journeys)
    lc = last_click(real_journeys)

    # Both must have channel and attributed_revenue (the canonical join keys).
    required = {"channel", "attributed_revenue"}
    assert required.issubset(set(mk.columns))
    assert required.issubset(set(lc.columns))

    # Markov should cover every channel observed in the data.
    input_channels = set(real_journeys["channel"].unique())
    output_channels = set(mk["channel"])
    assert input_channels == output_channels