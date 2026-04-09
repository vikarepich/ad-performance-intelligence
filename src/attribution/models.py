"""
Attribution Models — 5 models to distribute conversion credit across touchpoints.

What this module does:
- Takes user journeys (sequences of touchpoints)
- Applies 5 different attribution models
- Each model distributes the conversion revenue differently

The big question:
If a user saw TikTok → Google Search → Meta Retargeting → Direct → Buy,
which channels actually CAUSED the purchase?

5 models, 5 different answers:

1. Last-click:  Direct gets 100% (what GA4 does by default — misleading!)
2. First-click: TikTok gets 100% (who started the journey)
3. Linear:      Each gets 25% (everyone contributed equally)
4. Time-decay:  Direct 40%, Meta 30%, Google 20%, TikTok 10% (recency bias)
5. Shapley:     Mathematically fair distribution based on marginal contribution

Why Shapley is the best:
It comes from game theory (same math as SHAP in our ML models).
It asks: "If we REMOVE this channel, how much revenue do we lose?"
Channels that appear in many converting journeys get more credit.
"""

import pandas as pd
import numpy as np
from itertools import combinations


# ============================================================
# LAST-CLICK ATTRIBUTION
# ============================================================

def last_click(journeys_df):
    """
    Last-click attribution — 100% credit to the last touchpoint.

    This is the default in most analytics tools (GA4, Adobe).
    It's simple but deeply flawed: it ignores everything that
    happened before the final click.

    Result: bottom-funnel channels (brand search, direct) get
    all the credit. Top-funnel channels (TikTok, Meta) look worthless.

    Parameters:
        journeys_df: DataFrame with user journeys

    Returns:
        DataFrame with channel and attributed revenue
    """
    # Get only converting touchpoints (last touch of converting users)
    converting = journeys_df[journeys_df["is_converting_touch"] == True].copy()

    # All revenue goes to the last channel
    attribution = converting.groupby("channel").agg(
        attributed_revenue=("revenue", "sum"),
        conversions=("revenue", "count"),
    ).reset_index()

    attribution["model"] = "last_click"
    return attribution


# ============================================================
# FIRST-CLICK ATTRIBUTION
# ============================================================

def first_click(journeys_df):
    """
    First-click attribution — 100% credit to the first touchpoint.

    The opposite of last-click: rewards the channel that
    INTRODUCED the user to the brand.

    Useful for understanding awareness, but ignores nurturing.
    """
    # Get converting users
    converting_users = journeys_df[
        journeys_df["is_converting_touch"] == True
    ]["user_id"].unique()

    # Get first touchpoint of each converting user
    converting_journeys = journeys_df[
        journeys_df["user_id"].isin(converting_users)
    ]

    first_touches = converting_journeys.sort_values(
        ["user_id", "touch_index"]
    ).groupby("user_id").first().reset_index()

    # Get revenue from last touch (where conversion happened)
    revenues = journeys_df[
        journeys_df["is_converting_touch"] == True
    ][["user_id", "revenue"]]

    # Merge: first channel gets the revenue
    first_with_revenue = first_touches[["user_id", "channel"]].merge(
        revenues, on="user_id"
    )

    attribution = first_with_revenue.groupby("channel").agg(
        attributed_revenue=("revenue", "sum"),
        conversions=("revenue", "count"),
    ).reset_index()

    attribution["model"] = "first_click"
    return attribution


# ============================================================
# LINEAR ATTRIBUTION
# ============================================================

def linear(journeys_df):
    """
    Linear attribution — equal credit to all touchpoints.

    If a user had 4 touchpoints and spent $100,
    each channel gets $25. Simple and fair, but treats
    an awareness ad the same as a purchase-intent click.
    """
    # Get converting users and their revenue
    converting_touches = journeys_df[
        journeys_df["is_converting_touch"] == True
    ][["user_id", "revenue"]]

    # Get all touchpoints of converting users
    converting_users = converting_touches["user_id"].unique()
    all_touches = journeys_df[
        journeys_df["user_id"].isin(converting_users)
    ].copy()

    # Count touchpoints per user
    touch_counts = all_touches.groupby("user_id").size().reset_index(name="n_touches")

    # Merge to get revenue and touch count
    all_touches = all_touches.merge(touch_counts, on="user_id")
    all_touches = all_touches.merge(converting_touches, on="user_id", suffixes=("", "_total"))

    # Each touchpoint gets equal share: revenue / n_touches
    all_touches["attributed_revenue"] = all_touches["revenue_total"] / all_touches["n_touches"]

    attribution = all_touches.groupby("channel").agg(
        attributed_revenue=("attributed_revenue", "sum"),
        conversions=("user_id", "nunique"),
    ).reset_index()

    attribution["model"] = "linear"
    return attribution


# ============================================================
# TIME-DECAY ATTRIBUTION
# ============================================================

def time_decay(journeys_df, decay_rate=0.5):
    """
    Time-decay attribution — more credit to touchpoints closer to conversion.

    Uses exponential decay: weight = decay_rate ^ (days_before_conversion).
    Touchpoints right before purchase get the most credit.
    Earlier touchpoints get exponentially less.

    This makes intuitive sense: the retargeting ad yesterday
    probably mattered more than the TikTok video 3 weeks ago.

    Parameters:
        journeys_df: DataFrame with user journeys
        decay_rate: how fast credit decays (0.5 = halves each day before)
    """
    converting_touches = journeys_df[
        journeys_df["is_converting_touch"] == True
    ][["user_id", "revenue", "date"]].rename(columns={"date": "conversion_date"})

    converting_users = converting_touches["user_id"].unique()
    all_touches = journeys_df[
        journeys_df["user_id"].isin(converting_users)
    ].copy()

    # Merge conversion date
    all_touches = all_touches.merge(
        converting_touches[["user_id", "conversion_date", "revenue"]],
        on="user_id",
        suffixes=("", "_total"),
    )

    # Calculate days before conversion
    all_touches["date"] = pd.to_datetime(all_touches["date"])
    all_touches["conversion_date"] = pd.to_datetime(all_touches["conversion_date"])
    all_touches["days_before"] = (
        all_touches["conversion_date"] - all_touches["date"]
    ).dt.days

    # Calculate decay weight
    all_touches["weight"] = decay_rate ** all_touches["days_before"]

    # Normalize weights per user (so they sum to 1)
    weight_sums = all_touches.groupby("user_id")["weight"].transform("sum")
    all_touches["weight_normalized"] = all_touches["weight"] / weight_sums

    # Distribute revenue by weight
    all_touches["attributed_revenue"] = (
        all_touches["revenue_total"] * all_touches["weight_normalized"]
    )

    attribution = all_touches.groupby("channel").agg(
        attributed_revenue=("attributed_revenue", "sum"),
        conversions=("user_id", "nunique"),
    ).reset_index()

    attribution["model"] = "time_decay"
    return attribution


# ============================================================
# SHAPLEY ATTRIBUTION
# ============================================================

def shapley(journeys_df, max_users=500):
    """
    Shapley attribution — mathematically fair credit distribution.

    Based on Shapley values from cooperative game theory:
    "What is each channel's marginal contribution to conversions?"

    How it works:
    1. Look at all possible subsets of channels
    2. For each channel, calculate: "If we ADD this channel,
       how much does the conversion rate increase?"
    3. Average this across all possible orderings

    This is the same math behind SHAP (our ML explainer).
    It's computationally expensive but the most theoretically sound.

    Parameters:
        journeys_df: DataFrame with user journeys
        max_users: limit users for computation speed
    """
    # Get converting users and their journeys
    converting_users = journeys_df[
        journeys_df["is_converting_touch"] == True
    ]["user_id"].unique()

    # Limit for computation speed
    if len(converting_users) > max_users:
        rng = np.random.default_rng(42)
        converting_users = rng.choice(converting_users, max_users, replace=False)

    # Build channel sets per user
    user_channels = {}
    user_revenue = {}

    for user_id in converting_users:
        user_data = journeys_df[journeys_df["user_id"] == user_id]
        channels = tuple(sorted(user_data["channel"].unique()))
        revenue = user_data[user_data["is_converting_touch"] == True]["revenue"].sum()
        user_channels[user_id] = channels
        user_revenue[user_id] = revenue

    # Get all unique channels
    all_channels = sorted(set(
        ch for channels in user_channels.values() for ch in channels
    ))

    # Calculate conversion rate for each subset of channels
    # A "coalition" is a subset of channels that could have touched the user
    def coalition_value(coalition):
        """Revenue from users whose channels are a subset of this coalition."""
        coalition_set = set(coalition)
        total_revenue = 0
        count = 0
        for user_id, channels in user_channels.items():
            if set(channels).issubset(coalition_set):
                total_revenue += user_revenue[user_id]
                count += 1
        return total_revenue

    # Calculate Shapley value for each channel
    n = len(all_channels)
    shapley_values = {ch: 0.0 for ch in all_channels}

    # For each channel, calculate marginal contribution across all permutations
    # Simplified: iterate over all subsets (exact for small n, sampled for large)
    for channel in all_channels:
        other_channels = [c for c in all_channels if c != channel]

        # For each possible subset size
        for size in range(len(other_channels) + 1):
            for subset in combinations(other_channels, size):
                # Value with this channel
                with_channel = coalition_value(list(subset) + [channel])
                # Value without this channel
                without_channel = coalition_value(list(subset))
                # Marginal contribution
                marginal = with_channel - without_channel

                # Shapley weight: |S|!(n-|S|-1)! / n!
                s = len(subset)
                weight = (
                    np.math.factorial(s) * np.math.factorial(n - s - 1)
                    / np.math.factorial(n)
                )

                shapley_values[channel] += weight * marginal

    # Build attribution DataFrame
    attribution = pd.DataFrame([
        {
            "channel": channel,
            "attributed_revenue": round(value, 2),
            "conversions": sum(
                1 for channels in user_channels.values() if channel in channels
            ),
        }
        for channel, value in shapley_values.items()
    ])

    attribution["model"] = "shapley"
    return attribution


# ============================================================
# RUN ALL MODELS
# ============================================================

def run_all_models(journeys_df):
    """
    Run all 5 attribution models and return combined results.

    Returns:
        DataFrame with all models' attributions stacked
    """
    print("Running attribution models...")

    results = []

    print("  1/5 Last-click...")
    results.append(last_click(journeys_df))

    print("  2/5 First-click...")
    results.append(first_click(journeys_df))

    print("  3/5 Linear...")
    results.append(linear(journeys_df))

    print("  4/5 Time-decay...")
    results.append(time_decay(journeys_df))

    print("  5/5 Shapley (this takes a moment)...")
    results.append(shapley(journeys_df))

    combined = pd.concat(results, ignore_index=True)
    print(f"\nAttribution complete: {len(combined)} rows across 5 models")

    return combined