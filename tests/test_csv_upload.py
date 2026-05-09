"""
Unit tests for src/connectors/csv_upload.py.

Test strategy
-------------
- Each platform gets a mini-CSV fixture with its own column conventions.
- Detection tests verify platform identification works for all 4 platforms.
- find_column has a dedicated regression test for the
  'specific-over-generic' bug (Conv. vs Conv. value).
- End-to-end tests cover the most important real-world scenarios:
  Google Ads basic mapping, Meta ROAS edge case, unknown CSV warnings.
- Persistence tests use pytest's tmp_path fixture — never touch
  the real data/raw/ folder.

Run from project root:  pytest tests/test_csv_upload.py -v
"""
import io
from pathlib import Path

import pandas as pd
import pytest

from src.connectors.csv_upload import (
    detect_platform,
    find_column,
    map_columns,
    process_uploaded_csv,
    save_uploaded_data,
)


# ============================================================
# FIXTURES — mini CSVs in StringIO, one per platform
# ============================================================

@pytest.fixture
def google_ads_csv():
    """Realistic-looking Google Ads CSV export."""
    return io.StringIO(
        "Campaign,Day,Impr.,Clicks,Cost,Conv.,Conv. value\n"
        "Brand,2024-01-01,1000,50,25.50,5,150.00\n"
        "Brand,2024-01-02,1200,60,30.00,6,180.00\n"
        "Generic,2024-01-01,800,40,20.00,4,120.00\n"
    )


@pytest.fixture
def meta_ads_csv():
    """Meta Ads with ROAS column (instead of revenue) — needs multiplication."""
    return io.StringIO(
        "Campaign Name,Day,Impressions,Link clicks,Amount spent,Results,Purchase ROAS\n"
        "Awareness,2024-01-01,5000,200,100.00,15,3.0\n"
        "Conversion,2024-01-01,3000,150,80.00,20,4.5\n"
    )


@pytest.fixture
def tiktok_ads_csv():
    """TikTok Ads with its own column conventions."""
    return io.StringIO(
        "Campaign Name,Date,Impression,Click,Cost,Conversion,Total complete payment amount\n"
        "Discovery,2024-01-01,8000,400,60.00,10,250.00\n"
        "Conversion,2024-01-01,5000,300,75.00,15,400.00\n"
    )


@pytest.fixture
def ga4_csv():
    """GA4 export — no spend tracking, key events."""
    return io.StringIO(
        "Session campaign,Date,Sessions,Engaged sessions,Key events,Purchase revenue\n"
        "summer_promo,2024-01-01,1000,500,30,500.00\n"
        "winter_promo,2024-01-01,800,400,25,400.00\n"
    )


@pytest.fixture
def unknown_csv():
    """A CSV that doesn't match any platform signature."""
    return io.StringIO(
        "weird_col_a,weird_col_b,weird_col_c\n"
        "x,y,z\n"
        "1,2,3\n"
    )


# ============================================================
# detect_platform
# ============================================================

def test_detect_google_ads(google_ads_csv):
    df = pd.read_csv(google_ads_csv)
    assert detect_platform(df) == "google_ads"


def test_detect_meta_ads(meta_ads_csv):
    df = pd.read_csv(meta_ads_csv)
    assert detect_platform(df) == "meta_ads"


def test_detect_tiktok_ads(tiktok_ads_csv):
    df = pd.read_csv(tiktok_ads_csv)
    assert detect_platform(df) == "tiktok_ads"


def test_detect_ga4(ga4_csv):
    df = pd.read_csv(ga4_csv)
    assert detect_platform(df) == "ga4"


def test_detect_unknown_returns_unknown(unknown_csv):
    df = pd.read_csv(unknown_csv)
    assert detect_platform(df) == "unknown"


# ============================================================
# find_column — REGRESSION TEST for the Conv. vs Conv. value bug
# ============================================================

def test_find_column_prefers_specific_over_generic():
    """Regression test: find_column must NOT match 'Conv.' when looking
    for 'Conv. value'.

    Bug we found and fixed: substring-in-both-directions matching caused
    'Conv.' (the column for raw conversions) to win over 'Conv. value'
    (the column for monetary value). This re-routed revenue to a count
    field, silently producing wrong numbers downstream.

    Fix: prefer exact match in pass 1; one-way partial match only in
    pass 2. This test enforces the fix.
    """
    cols_with_both = [
        ("Campaign",   "campaign"),
        ("Conv.",      "conv."),
        ("Conv. value", "conv. value"),
    ]

    # We're looking for "conv. value" — must return the value column,
    # NOT the count column.
    found = find_column(cols_with_both, ["conv. value"])
    assert found == "Conv. value", (
        f"Expected 'Conv. value', got {found!r} — short generic name "
        "wrongly preferred over specific name."
    )


# ============================================================
# process_uploaded_csv — end-to-end with real-data scenarios
# ============================================================

def test_process_google_ads_revenue_correctly_mapped(google_ads_csv):
    """Smoke test on Google Ads: every key column lands in the right place,
    and revenue is the value column (not the count column)."""
    result = process_uploaded_csv(google_ads_csv)

    assert result["platform"] == "google_ads"
    assert result["row_count"] == 3
    # The 'orders' column (All conv. in Google Ads) isn't in our fixture,
    # so the mapper correctly warns about it. We assert specifically that
    # this is the ONLY warning — anything else means a regression.
    assert result["warnings"] == ["Column 'orders' not found in CSV — set to 0"]

    df = result["data"]
    # Revenue must be the monetary value, not the count.
    assert df["revenue"].tolist() == [150.0, 180.0, 120.0]
    # Spend must be Cost.
    assert df["mark_spent"].tolist() == [25.5, 30.0, 20.0]
    # Source tag and category must be populated for downstream pipeline.
    assert (df["source"] == "google_ads").all()
    assert (df["category"] == "search").all()


def test_process_meta_ads_roas_multiplied_by_spend(meta_ads_csv):
    """Meta exports often give 'Purchase ROAS' instead of revenue.
    The mapper must detect this and multiply by spend to recover revenue.

    Row 1: spend=100, ROAS=3.0   -> revenue should be 300.0
    Row 2: spend=80,  ROAS=4.5   -> revenue should be 360.0
    """
    result = process_uploaded_csv(meta_ads_csv)
    assert result["platform"] == "meta_ads"

    df = result["data"]
    assert df["revenue"].tolist() == [300.0, 360.0]
    assert df["mark_spent"].tolist() == [100.0, 80.0]


def test_process_unknown_platform_returns_warnings(unknown_csv):
    """An unrecognised CSV must not crash — instead return an explanatory
    warning and the original column list so the user can pick manually."""
    result = process_uploaded_csv(unknown_csv)

    assert result["platform"] == "unknown"
    assert result["data"] is None
    assert len(result["warnings"]) > 0
    assert "weird_col_a" in result["original_columns"]


# ============================================================
# save_uploaded_data — uses tmp_path, NEVER writes to data/raw/
# ============================================================

def test_save_uploaded_data_dedupes_on_append(monkeypatch, tmp_path):
    """When appending the same data twice, duplicates on
    (source, campaign_name, c_date) must be removed — keeping the last one.

    We monkeypatch the output path so this test never touches
    the real data/raw/ folder.
    """
    # Redirect the hard-coded output path to a tmp directory.
    fake_path = tmp_path / "combined_campaigns.csv"

    import src.connectors.csv_upload as csv_upload_module
    real_path_class = csv_upload_module.Path

    class FakePath(type(fake_path)):
        def __new__(cls, *args, **kwargs):
            if args and args[0] == "data/raw/combined_campaigns.csv":
                return fake_path
            return real_path_class(*args, **kwargs)

    monkeypatch.setattr(csv_upload_module, "Path", FakePath)

    # First save — 2 rows.
    df_initial = pd.DataFrame({
        "source": ["google_ads"] * 2,
        "campaign_name": ["A", "B"],
        "c_date": ["2024-01-01", "2024-01-01"],
        "revenue": [100.0, 200.0],
    })
    save_uploaded_data(df_initial, append=True)
    assert fake_path.exists()
    assert len(pd.read_csv(fake_path)) == 2

    # Second save — same 2 rows. After dedup the file must still have 2 rows.
    save_uploaded_data(df_initial, append=True)
    final = pd.read_csv(fake_path)
    assert len(final) == 2, f"Expected 2 rows after dedup, got {len(final)}"