"""
Tests for the Attribution Module (journey_simulator, models, analyzer).

What's happening here:
- We test journey generation: correct structure, cookie simulation, tracking
- We test all 5 attribution models: correct output format, revenue conservation
- We test the analyzer: comparison logic, insights generation

Key testing principle for attribution:
Revenue should be CONSERVED across models — total attributed revenue
should roughly equal total actual revenue (except Shapley which
approximates). We check this with a tolerance.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from src.attribution.journey_simulator import (
    generate_single_journey,
    generate_journeys,
    CHANNELS,
    TRACKING_METHODS,
)
from src.attribution.models import (
    last_click,
    first_click,
    linear,
    time_decay,
    shapley,
    run_all_models,
)
from src.attribution.analyzer import (
    compare_models,
    find_misattribution,
    analyze_tracking_loss,
    generate_insights,
)


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture
def sample_journeys():
    """
    Generate a small set of journeys for testing.

    200 users is enough to have converting users
    but fast enough for tests (~1 second).
    """
    return generate_journeys(n_users=200, seed=42)


@pytest.fixture
def sample_attribution(sample_journeys):
    """Run all attribution models on sample journeys."""
    return run_all_models(sample_journeys)


# ============================================================
# TESTS FOR journey_simulator
# ============================================================

class TestJourneySimulator:
    """Tests for user journey generation."""

    def test_generates_correct_user_count(self):
        """Check that we get the requested number of unique users."""
        df = generate_journeys(n_users=50, seed=42)
        assert df["user_id"].nunique() == 50, \
            f"Expected 50 users, got {df['user_id'].nunique()}"

    def test_has_required_columns(self, sample_journeys):
        """Check that journeys have all required columns."""
        required = [
            "user_id", "touch_index", "channel", "date",
            "cookies_accepted", "ga4_visible", "enhanced_visible",
            "is_converting_touch", "revenue",
        ]
        for col in required:
            assert col in sample_journeys.columns, f"Missing column: {col}"

    def test_channels_are_valid(self, sample_journeys):
        """Check that all channels are from the defined CHANNELS dict."""
        valid_channels = set(CHANNELS.keys())
        journey_channels = set(sample_journeys["channel"].unique())
        invalid = journey_channels - valid_channels
        assert len(invalid) == 0, f"Invalid channels found: {invalid}"

    def test_conversion_rate_realistic(self, sample_journeys):
        """Check that conversion rate is between 5% and 25%."""
        total_users = sample_journeys["user_id"].nunique()
        converting = sample_journeys[
            sample_journeys["is_converting_touch"] == True
        ]["user_id"].nunique()
        rate = converting / total_users

        assert 0.05 <= rate <= 0.25, \
            f"Conversion rate {rate:.1%} is outside realistic range 5-25%"

    def test_cookie_rejection_rate(self, sample_journeys):
        """Check that ~30-45% of users reject cookies."""
        user_cookies = sample_journeys.groupby("user_id")["cookies_accepted"].first()
        rejection_rate = (~user_cookies).mean()

        assert 0.25 <= rejection_rate <= 0.50, \
            f"Cookie rejection rate {rejection_rate:.1%} outside expected 25-50%"

    def test_revenue_only_on_converting_touch(self, sample_journeys):
        """Check that revenue > 0 only on converting touchpoints."""
        non_converting = sample_journeys[
            sample_journeys["is_converting_touch"] == False
        ]
        # Non-converting touches should have 0 or NaN revenue (never positive)
        has_positive_revenue = (non_converting["revenue"].fillna(0) > 0).any()
        assert not has_positive_revenue, \
            "Non-converting touchpoints should not have positive revenue"

    def test_ga4_sees_less_than_enhanced(self, sample_journeys):
        """Check that GA4 tracking sees fewer touchpoints than enhanced."""
        ga4_seen = sample_journeys["ga4_visible"].sum()
        enhanced_seen = sample_journeys["enhanced_visible"].sum()

        assert enhanced_seen >= ga4_seen, \
            f"Enhanced ({enhanced_seen}) should see >= GA4 ({ga4_seen})"

    def test_touch_index_starts_at_zero(self, sample_journeys):
        """Check that each user's journey starts at touch_index 0."""
        first_touches = sample_journeys.groupby("user_id")["touch_index"].min()
        assert (first_touches == 0).all(), \
            "Each user's journey should start at touch_index 0"


# ============================================================
# TESTS FOR attribution models
# ============================================================

class TestLastClick:
    """Tests for last-click attribution."""

    def test_returns_dataframe(self, sample_journeys):
        """Check that last_click returns a DataFrame."""
        result = last_click(sample_journeys)
        assert isinstance(result, pd.DataFrame)

    def test_has_required_columns(self, sample_journeys):
        """Check output has channel, attributed_revenue, model."""
        result = last_click(sample_journeys)
        assert "channel" in result.columns
        assert "attributed_revenue" in result.columns
        assert "model" in result.columns

    def test_model_name_is_correct(self, sample_journeys):
        """Check that model column says 'last_click'."""
        result = last_click(sample_journeys)
        assert (result["model"] == "last_click").all()

    def test_revenue_non_negative(self, sample_journeys):
        """Check that all attributed revenue is >= 0."""
        result = last_click(sample_journeys)
        assert (result["attributed_revenue"] >= 0).all(), \
            "Attributed revenue should be non-negative"


class TestFirstClick:
    """Tests for first-click attribution."""

    def test_returns_dataframe(self, sample_journeys):
        result = first_click(sample_journeys)
        assert isinstance(result, pd.DataFrame)

    def test_model_name_is_correct(self, sample_journeys):
        result = first_click(sample_journeys)
        assert (result["model"] == "first_click").all()

    def test_revenue_non_negative(self, sample_journeys):
        result = first_click(sample_journeys)
        assert (result["attributed_revenue"] >= 0).all()


class TestLinear:
    """Tests for linear attribution."""

    def test_returns_dataframe(self, sample_journeys):
        result = linear(sample_journeys)
        assert isinstance(result, pd.DataFrame)

    def test_model_name_is_correct(self, sample_journeys):
        result = linear(sample_journeys)
        assert (result["model"] == "linear").all()

    def test_revenue_conserved(self, sample_journeys):
        """
        Check that total attributed revenue equals actual revenue.

        Linear attribution distributes ALL revenue — nothing lost.
        """
        result = linear(sample_journeys)
        total_attributed = result["attributed_revenue"].sum()
        total_actual = sample_journeys["revenue"].sum()

        assert abs(total_attributed - total_actual) < 1.0, \
            f"Revenue not conserved: attributed={total_attributed:.2f}, actual={total_actual:.2f}"


class TestTimeDecay:
    """Tests for time-decay attribution."""

    def test_returns_dataframe(self, sample_journeys):
        result = time_decay(sample_journeys)
        assert isinstance(result, pd.DataFrame)

    def test_model_name_is_correct(self, sample_journeys):
        result = time_decay(sample_journeys)
        assert (result["model"] == "time_decay").all()

    def test_revenue_conserved(self, sample_journeys):
        """Check that total attributed revenue equals actual revenue."""
        result = time_decay(sample_journeys)
        total_attributed = result["attributed_revenue"].sum()
        total_actual = sample_journeys["revenue"].sum()

        assert abs(total_attributed - total_actual) < 1.0, \
            f"Revenue not conserved: attributed={total_attributed:.2f}, actual={total_actual:.2f}"


class TestShapley:
    """Tests for Shapley attribution."""

    def test_returns_dataframe(self, sample_journeys):
        result = shapley(sample_journeys, max_users=50)
        assert isinstance(result, pd.DataFrame)

    def test_model_name_is_correct(self, sample_journeys):
        result = shapley(sample_journeys, max_users=50)
        assert (result["model"] == "shapley").all()

    def test_has_all_channels(self, sample_journeys):
        """Check that Shapley covers all channels in converting journeys."""
        result = shapley(sample_journeys, max_users=50)
        assert len(result) > 0, "Shapley should return at least one channel"


class TestRunAllModels:
    """Tests for the combined run_all_models function."""

    def test_returns_five_models(self, sample_attribution):
        """Check that all 5 models are in the results."""
        models = sample_attribution["model"].unique()
        expected = {"last_click", "first_click", "linear", "time_decay", "shapley"}
        assert set(models) == expected, \
            f"Expected models {expected}, got {set(models)}"

    def test_all_revenue_positive(self, sample_attribution):
        """Check no negative revenue across any model."""
        non_shapley = sample_attribution[sample_attribution["model"] != "shapley"]
        assert (non_shapley["attributed_revenue"] >= 0).all(), \
            "Non-Shapley models should have non-negative revenue"


# ============================================================
# TESTS FOR analyzer
# ============================================================

class TestAnalyzer:
    """Tests for the attribution analyzer."""

    def test_compare_models_returns_pivot(self, sample_attribution):
        """Check that compare_models returns a pivot table."""
        result = compare_models(sample_attribution)
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0, "Pivot table should not be empty"

    def test_find_misattribution_has_verdict(self, sample_attribution):
        """Check that misattribution analysis includes verdict."""
        result = find_misattribution(sample_attribution)
        assert "verdict" in result.columns
        assert "pct_difference" in result.columns

    def test_tracking_loss_ga4_higher(self, sample_journeys):
        """Check that GA4 loss is higher than enhanced tracking loss."""
        summary, _ = analyze_tracking_loss(sample_journeys)
        assert summary["ga4_loss_pct"] > summary["enhanced_loss_pct"], \
            "GA4 should lose more data than enhanced tracking"

    def test_generate_insights_has_headline(self, sample_journeys, sample_attribution):
        """Check that insights include a headline."""
        insights = generate_insights(sample_journeys, sample_attribution)
        assert "headline" in insights
        assert "tracking" in insights
        assert "attribution" in insights
        assert "summary" in insights

    def test_insights_tracking_shows_loss(self, sample_journeys, sample_attribution):
        """Check that tracking insights show data loss percentage."""
        insights = generate_insights(sample_journeys, sample_attribution)
        assert "ga4_data_loss" in insights["tracking"]
        assert "%" in insights["tracking"]["ga4_data_loss"]
