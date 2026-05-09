"""
Attribution Analyzer — compares attribution models and reveals insights.

What this module does:
- Runs all 5 attribution models on user journey data
- Compares how differently each model allocates revenue
- Shows which channels are overvalued/undervalued by last-click
- Calculates GA4 data loss vs enhanced tracking
- Saves results for dashboard visualization

Key insight for portfolio:
"Last-click says Google Brand drives 45% of revenue.
Shapley shows TikTok initiates 30% of converting journeys
but gets 0% credit under last-click. This is a $X misallocation."
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

from src.attribution.journey_simulator import generate_journeys, save_journeys
from src.attribution.models import run_all_models

# ============================================================
# PATHS
# ============================================================

JOURNEYS_PATH = Path("data/processed/user_journeys.csv")
ATTRIBUTION_PATH = Path("data/processed/attribution_results.csv")
INSIGHTS_PATH = Path("models/attribution_insights.json")


# ============================================================
# ANALYSIS FUNCTIONS
# ============================================================

def compare_models(attribution_df):
    """
    Compare revenue allocation across all attribution models.

    Creates a pivot table:
    - Rows = channels
    - Columns = models
    - Values = attributed revenue

    This makes it easy to see: "TikTok gets $0 under last-click
    but $5,000 under Shapley — that's a massive blind spot."
    """
    pivot = attribution_df.pivot_table(
        index="channel",
        columns="model",
        values="attributed_revenue",
        aggfunc="sum",
    ).fillna(0).round(2)

    # Add percentage columns
    for col in pivot.columns:
        total = pivot[col].sum()
        if total > 0:
            pivot[f"{col}_pct"] = (pivot[col] / total * 100).round(1)

    return pivot


def find_misattribution(attribution_df):
    """
    Find channels where last-click disagrees with data-driven models
    (Shapley and Markov).

    The money insight: channels that last-click systematically over- or
    undervalues compared to data-driven attribution. Both Shapley
    (cooperative game theory) and Markov (chain removal effect) are
    independent methodologies — when they agree on the direction of
    disagreement with last-click, that's a high-confidence signal.

    Returns a DataFrame with one row per channel, columns for each
    model's revenue and share, plus two verdicts: one Shapley vs
    last-click, one Markov vs last-click.
    """
    def _slice(model_name: str, revenue_col: str) -> pd.DataFrame:
        return (
            attribution_df[attribution_df["model"] == model_name]
            [["channel", "attributed_revenue"]]
            .rename(columns={"attributed_revenue": revenue_col})
        )

    last = _slice("last_click", "last_click_revenue")
    shap = _slice("shapley",    "shapley_revenue")
    mark = _slice("markov",     "markov_revenue")

    comparison = (
        last
        .merge(shap, on="channel", how="outer")
        .merge(mark, on="channel", how="outer")
        .fillna(0)
    )

    # Calculate share-of-total for each model.
    for col_revenue, col_pct in [
        ("last_click_revenue", "last_click_pct"),
        ("shapley_revenue",    "shapley_pct"),
        ("markov_revenue",     "markov_pct"),
    ]:
        total = comparison[col_revenue].sum()
        comparison[col_pct] = (
            (comparison[col_revenue] / total * 100).round(1)
            if total > 0 else 0
        )

    # Difference vs last-click for each data-driven model.
    comparison["shapley_vs_lc_pp"] = (
        comparison["shapley_pct"] - comparison["last_click_pct"]
    ).round(1)
    comparison["markov_vs_lc_pp"] = (
        comparison["markov_pct"] - comparison["last_click_pct"]
    ).round(1)

    # Backward-compat: keep the old column name for any downstream consumer
    # (Streamlit, MCP server, notebooks) that already references it.
    comparison["pct_difference"] = comparison["shapley_vs_lc_pp"]

    def _verdict(diff_pp: float) -> str:
        if diff_pp > 3:
            return "UNDERVALUED by last-click"
        if diff_pp < -3:
            return "OVERVALUED by last-click"
        return "Fairly valued"

    comparison["verdict"]        = comparison["shapley_vs_lc_pp"].apply(_verdict)
    comparison["verdict_markov"] = comparison["markov_vs_lc_pp"].apply(_verdict)

    # Highlight channels where both data-driven models agree on the
    # direction of disagreement with last-click — the highest-confidence
    # signal for a budget reallocation conversation.
    comparison["both_models_agree"] = (
        (comparison["shapley_vs_lc_pp"] > 3) & (comparison["markov_vs_lc_pp"] > 3)
    ) | (
        (comparison["shapley_vs_lc_pp"] < -3) & (comparison["markov_vs_lc_pp"] < -3)
    )

    return comparison.sort_values("shapley_vs_lc_pp", ascending=False)

def analyze_tracking_loss(journeys_df):
    """
    Calculate how much data GA4 loses vs enhanced tracking.

    This shows the business impact of cookie rejection:
    "GA4 misses 35% of touchpoints → wrong attribution → budget waste."
    """
    total = len(journeys_df)
    ga4_seen = journeys_df["ga4_visible"].sum()
    enhanced_seen = journeys_df["enhanced_visible"].sum()

    # Per-channel breakdown
    channel_tracking = journeys_df.groupby("channel").agg(
        total_touches=("ga4_visible", "count"),
        ga4_visible=("ga4_visible", "sum"),
        enhanced_visible=("enhanced_visible", "sum"),
    ).reset_index()

    channel_tracking["ga4_rate"] = (
        channel_tracking["ga4_visible"] / channel_tracking["total_touches"] * 100
    ).round(1)
    channel_tracking["enhanced_rate"] = (
        channel_tracking["enhanced_visible"] / channel_tracking["total_touches"] * 100
    ).round(1)
    channel_tracking["ga4_loss"] = (100 - channel_tracking["ga4_rate"]).round(1)
    channel_tracking["recovered_by_enhanced"] = (
        channel_tracking["enhanced_rate"] - channel_tracking["ga4_rate"]
    ).round(1)

    summary = {
        "total_touchpoints": total,
        "ga4_visible": int(ga4_seen),
        "enhanced_visible": int(enhanced_seen),
        "ga4_loss_pct": round((1 - ga4_seen / total) * 100, 1),
        "enhanced_loss_pct": round((1 - enhanced_seen / total) * 100, 1),
        "recovered_pct": round((enhanced_seen - ga4_seen) / total * 100, 1),
    }

    return summary, channel_tracking


def generate_insights(journeys_df, attribution_df):
    """
    Generate key insights as JSON for dashboard and portfolio.
    """
    # Tracking analysis
    tracking_summary, channel_tracking = analyze_tracking_loss(journeys_df)

    # Misattribution analysis
    misattribution = find_misattribution(attribution_df)

    # Find most undervalued channel
    most_undervalued = misattribution.iloc[0]
    most_overvalued = misattribution.iloc[-1]

    # Converting users stats
    total_users = journeys_df["user_id"].nunique()
    converting_users = journeys_df[
        journeys_df["is_converting_touch"] == True
    ]["user_id"].nunique()
    total_revenue = journeys_df["revenue"].sum()

    # Cookie stats
    cookie_rejection_rate = (
        ~journeys_df.groupby("user_id")["cookies_accepted"].first()
    ).mean() * 100

    insights = {
        "headline": (
            f"Last-click attribution misallocates revenue: "
            f"{most_undervalued['channel']} is undervalued by "
            f"{most_undervalued['pct_difference']:+.1f}pp, while "
            f"{most_overvalued['channel']} is overvalued by "
            f"{abs(most_overvalued['pct_difference']):.1f}pp."
        ),
        "tracking": {
            "ga4_data_loss": f"{tracking_summary['ga4_loss_pct']}%",
            "enhanced_recovery": f"{tracking_summary['recovered_pct']}%",
            "cookie_rejection_rate": f"{cookie_rejection_rate:.1f}%",
        },
        "attribution": {
            "most_undervalued": {
                "channel": most_undervalued["channel"],
                "last_click_pct": float(most_undervalued["last_click_pct"]),
                "shapley_pct": float(most_undervalued["shapley_pct"]),
                "difference": float(most_undervalued["pct_difference"]),
            },
            "most_overvalued": {
                "channel": most_overvalued["channel"],
                "last_click_pct": float(most_overvalued["last_click_pct"]),
                "shapley_pct": float(most_overvalued["shapley_pct"]),
                "difference": float(most_overvalued["pct_difference"]),
            },
        },
        "summary": {
            "total_users": total_users,
            "converting_users": converting_users,
            "conversion_rate": f"{converting_users / total_users * 100:.1f}%",
            "total_revenue": round(float(total_revenue), 2),
        },
    }

    return insights


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(attribution_df, insights):
    """Save attribution results and insights."""
    ATTRIBUTION_PATH.parent.mkdir(parents=True, exist_ok=True)
    INSIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    attribution_df.to_csv(ATTRIBUTION_PATH, index=False)
    print(f"Attribution results saved to {ATTRIBUTION_PATH}")

    with open(INSIGHTS_PATH, "w") as f:
        json.dump(insights, f, indent=2)
    print(f"Insights saved to {INSIGHTS_PATH}")


# ============================================================
# MAIN
# ============================================================

def run_analyzer():
    """
    Full attribution analysis pipeline:
    1. Generate user journeys
    2. Run all 5 attribution models
    3. Compare and analyze
    4. Save results
    """
    print("=" * 60)
    print("  MULTI-TOUCH ATTRIBUTION ANALYSIS")
    print("=" * 60)

    # Step 1: generate journeys
    journeys_df = generate_journeys(n_users=2000)
    save_journeys(journeys_df)

    # Step 2: run attribution models
    print(f"\n{'=' * 60}")
    attribution_df = run_all_models(journeys_df)

    # Step 3: compare models
    print(f"\n{'=' * 60}")
    print("  MODEL COMPARISON")
    print(f"{'=' * 60}")

    pivot = compare_models(attribution_df)
    print("\nRevenue attribution by model ($):")
    revenue_cols = [c for c in pivot.columns if not c.endswith("_pct")]
    print(pivot[revenue_cols].to_string())

    print("\nRevenue attribution by model (%):")
    pct_cols = [c for c in pivot.columns if c.endswith("_pct")]
    print(pivot[pct_cols].to_string())

    # Step 4: misattribution analysis
    print(f"\n{'=' * 60}")
    print("  MISATTRIBUTION: LAST-CLICK vs SHAPLEY")
    print(f"{'=' * 60}")

    misattribution = find_misattribution(attribution_df)
    print(misattribution[["channel", "last_click_pct", "shapley_pct",
                          "pct_difference", "verdict"]].to_string(index=False))

    # Step 5: tracking loss
    print(f"\n{'=' * 60}")
    print("  TRACKING DATA LOSS")
    print(f"{'=' * 60}")

    tracking_summary, channel_tracking = analyze_tracking_loss(journeys_df)
    print(f"\n  GA4 loses: {tracking_summary['ga4_loss_pct']}% of touchpoints")
    print(f"  Enhanced tracking recovers: {tracking_summary['recovered_pct']}% additional")
    print(f"\n  Per-channel tracking rates:")
    print(channel_tracking[["channel", "ga4_rate", "enhanced_rate",
                            "recovered_by_enhanced"]].to_string(index=False))

    # Step 6: generate and save insights
    insights = generate_insights(journeys_df, attribution_df)

    print(f"\n{'*' * 60}")
    print(f"  KEY INSIGHT:")
    print(f"  {insights['headline']}")
    print(f"{'*' * 60}")

    save_results(attribution_df, insights)

    return attribution_df, insights


if __name__ == "__main__":
    run_analyzer()