"""
Shapley Attribution — game-theoretic credit distribution.

Why Shapley deserves its own module:
- It's the only attribution model whose math comes from cooperative game
  theory (the same Shapley values that power SHAP in ML explainability).
- Computationally expensive (O(2^N) over channels) — keeping it isolated
  makes it easier to swap in approximations (sampled Shapley, KernelSHAP)
  without touching the simpler models.
- Tested as a standalone unit; analyzer.py + models.py both consume it.

Why Shapley vs Markov:
- Shapley: order-agnostic. Treats {TikTok, Google} as one coalition
  regardless of who came first. Best for revenue attribution.
- Markov:  order-sensitive. Models the customer journey as a probability
  graph. Best for conversion-rate attribution.
- This project ships BOTH so you can compare. When they agree → high
  confidence. When they disagree → the journey order matters and
  deserves investigation.
"""

import pandas as pd
import numpy as np
from itertools import combinations
from math import factorial


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
        for user_id, channels in user_channels.items():
            if set(channels).issubset(coalition_set):
                total_revenue += user_revenue[user_id]
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
                    factorial(s) * factorial(n - s - 1)
                    / factorial(n)
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