"""
Markov Chain Attribution — model touchpoints as states in a Markov chain.

What this module does:
- Converts user journeys into paths through a state graph
- Builds a transition matrix from observed paths
- Computes overall conversion probability
- For each channel: removes it, recomputes probability, measures the drop
- Normalizes those drops into channel weights ("attribution credit")

Why Markov vs Shapley:
- Markov is path-aware: 'TikTok → Google' is different from 'Google → TikTok'
- Shapley is order-agnostic: it treats {TikTok, Google} as one coalition
- Markov is O(N²) on channels — scales much better than Shapley's O(2^N)
- Both should agree directionally on most channels; Markov is more sensitive
  to journey order

Key business framing:
"Removal effect" of a channel = % drop in conversion probability if you
turned that channel off entirely. This is much closer to what marketing
managers actually want to know than 'last-click credit'.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Iterable

# ============================================================
# CONSTANTS
# ============================================================

# Special states in the Markov chain.
# - START: every journey begins here. Lets us model "first touch" probabilities.
# - CONVERSION: absorbing success state. User bought.
# - NULL:       absorbing failure state. User dropped out.
START = "START"
CONVERSION = "CONVERSION"
NULL = "NULL"


# ============================================================
# 1. BUILD PATHS  (DataFrame -> list of paths)
# ============================================================

def build_paths(journeys_df: pd.DataFrame) -> list[list[str]]:
    """
    Convert a journeys DataFrame into a list of paths through the Markov chain.

    Each path is a list of state names, always starting with START and
    ending with either CONVERSION (user bought) or NULL (user gave up).

    Example
    -------
    Input row sequence for one user:
        touch_index=0  channel=tiktok_paid              is_converting_touch=False
        touch_index=1  channel=meta_facebook_feed       is_converting_touch=False
        touch_index=2  channel=google_search_brand      is_converting_touch=True

    Output path:
        ['START', 'tiktok_paid', 'meta_facebook_feed', 'google_search_brand', 'CONVERSION']

    Parameters
    ----------
    journeys_df : DataFrame with columns user_id, touch_index, channel,
                  is_converting_touch.

    Returns
    -------
    list of paths (each path is a list of state names).
    """
    required = {"user_id", "touch_index", "channel", "is_converting_touch"}
    missing = required - set(journeys_df.columns)
    if missing:
        raise ValueError(f"build_paths: missing columns {missing}")

    # CRITICAL: sort by user_id, touch_index BEFORE groupby.
    # If we trusted the input order we'd be one bad upstream merge away
    # from a silent bug.
    df = journeys_df.sort_values(["user_id", "touch_index"], kind="stable")

    paths: list[list[str]] = []

    # groupby preserves the sort order we just established (stable sort).
    for _, user_data in df.groupby("user_id", sort=False):
        channels = user_data["channel"].tolist()
        # A user is a converter iff ANY of their touches is the converting one.
        # (In this dataset only the last touch can be converting, but we don't
        # want to depend on that assumption — it's brittle.)
        converted = bool(user_data["is_converting_touch"].any())

        end_state = CONVERSION if converted else NULL
        path = [START] + channels + [end_state]
        paths.append(path)

    return paths


# ============================================================
# 2. BUILD TRANSITION MATRIX  (paths -> P(to | from) matrix)
# ============================================================

def build_transition_matrix(paths: list[list[str]]) -> pd.DataFrame:
    """
    Compute P(to_state | from_state) from observed paths.

    Walks every consecutive pair (s_i, s_{i+1}) in every path, counts how
    often each transition occurred, and normalises each row to sum to 1.

    Absorbing states (CONVERSION, NULL) get a self-loop with probability 1
    by construction — once you're there, you stay there. This is the
    standard formulation for absorbing Markov chains.

    Parameters
    ----------
    paths : list of paths (each path = list of state names starting with
            START and ending with CONVERSION or NULL).

    Returns
    -------
    DataFrame with index = from_state, columns = to_state, values = P(to|from).
    Every row sums to 1.0. Row/column ordering: START, channels (sorted),
    CONVERSION, NULL — deterministic for reproducible tests.
    """
    if not paths:
        raise ValueError("build_transition_matrix: paths is empty")

    # ---- 1. Discover all unique channels (anything that's not START/CONV/NULL)
    channels = set()
    for path in paths:
        for state in path:
            if state not in (START, CONVERSION, NULL):
                channels.add(state)

    # Deterministic state ordering: START → channels (sorted) → CONVERSION → NULL.
    # This makes printing and testing predictable.
    states = [START] + sorted(channels) + [CONVERSION, NULL]
    n = len(states)
    state_to_idx = {s: i for i, s in enumerate(states)}

    # ---- 2. Count transitions
    # Use raw NumPy for speed; convert to DataFrame at the end.
    counts = np.zeros((n, n), dtype=np.float64)
    for path in paths:
        for s_from, s_to in zip(path[:-1], path[1:]):
            counts[state_to_idx[s_from], state_to_idx[s_to]] += 1

    # ---- 3. Normalise rows.
    # For non-absorbing states with zero outgoing transitions (rare/empty
    # in real data but possible in pathological inputs), leave the row as 0
    # to avoid division by zero. We'll patch absorbing rows separately below.
    row_sums = counts.sum(axis=1, keepdims=True)  # shape (n, 1)
    # np.divide with 'where' avoids the warning AND the NaN for zero rows.
    probs = np.divide(counts, row_sums, out=np.zeros_like(counts), where=row_sums > 0)

    # ---- 4. Make CONVERSION and NULL absorbing (self-loop = 1.0).
    # By construction, no path leaves these states, so their counts row
    # is all zeros — we explicitly install the self-loop here.
    probs[state_to_idx[CONVERSION], state_to_idx[CONVERSION]] = 1.0
    probs[state_to_idx[NULL], state_to_idx[NULL]] = 1.0

    return pd.DataFrame(probs, index=states, columns=states)

# ============================================================
# 3. CONVERSION PROBABILITY  (matrix -> P(reach CONVERSION from START))
# ============================================================

def conversion_probability(M: pd.DataFrame) -> float:
    """
    Compute the overall probability that a journey starting at START
    eventually reaches CONVERSION.

    Uses the absorbing Markov chain fundamental matrix:
        Split states into transient (T) and absorbing (A).
        Reorder M as [[Q, R], [0, I]] where:
            Q = transient → transient transitions (square)
            R = transient → absorbing transitions
        Then B = (I - Q)^(-1) @ R gives the absorption probabilities:
            B[t, a] = P(absorbed in state `a` | started in state `t`)
        Our answer is B[START, CONVERSION].

    Why this method:
        - Closed-form: one matrix inversion, no iteration.
        - Numerically stable when (I - Q) is well-conditioned (the typical case).
        - Falls back gracefully for degenerate inputs.

    Parameters
    ----------
    M : transition matrix DataFrame from build_transition_matrix.

    Returns
    -------
    float in [0, 1]: P(CONVERSION | start in START).
    """
    states = list(M.index)
    if START not in states or CONVERSION not in states:
        raise ValueError("Matrix must contain START and CONVERSION states")

    absorbing = {CONVERSION, NULL} & set(states)
    transient = [s for s in states if s not in absorbing]
    absorbing = [s for s in states if s in absorbing]  # preserve original order

    # Q: transient -> transient.  R: transient -> absorbing.
    Q = M.loc[transient, transient].values
    R = M.loc[transient, absorbing].values

    # Fundamental matrix N = (I - Q)^(-1).
    # If Q has eigenvalue 1 (e.g. an isolated cycle of transient states with
    # no exit), I - Q is singular and we fall back to iteration.
    n_t = len(transient)
    I = np.eye(n_t)
    try:
        N = np.linalg.inv(I - Q)
    except np.linalg.LinAlgError:
        # Iterative fallback: sum_{k=0..K} Q^k. K=200 is way past convergence
        # for any realistic marketing journey (avg length ~ 4-5 touches).
        N = I.copy()
        Qk = I.copy()
        for _ in range(200):
            Qk = Qk @ Q
            N = N + Qk

    B = N @ R  # shape (n_transient, n_absorbing)

    # Build a small DataFrame for readable indexing.
    B_df = pd.DataFrame(B, index=transient, columns=absorbing)
    return float(B_df.loc[START, CONVERSION])

# ============================================================
# 4. REMOVAL EFFECT  (matrix, channel -> drop in conversion prob)
# ============================================================

def removal_effect(M: pd.DataFrame, channel: str) -> float:
    """
    Compute the removal effect of a channel: the relative drop in overall
    conversion probability when this channel is "turned off".

    Interpretation: "if we couldn't run this channel anymore, we'd lose
    {removal_effect:.0%} of our conversions, all else equal."

    Algorithm:
        1. Compute baseline conversion probability P_base from M.
        2. Build M' = M with the channel's outgoing transitions replaced
           by a single transition to NULL with probability 1.
           (Standard formulation: removing a channel = users hitting it
           drop out, since the channel doesn't exist anymore.)
        3. Compute P_without = conversion_probability(M').
        4. Return (P_base - P_without) / P_base.

    Why redirect to NULL rather than delete the state:
        Deleting the state breaks the "rows sum to 1" invariant and makes
        the math ill-defined. Redirecting to NULL preserves probability
        conservation and matches the standard treatment in attribution
        literature.

    Parameters
    ----------
    M       : transition matrix from build_transition_matrix.
    channel : channel state name to remove.

    Returns
    -------
    float in [0, 1]: relative drop in conversion probability.
    Returns 0.0 if the channel doesn't exist in the matrix.
    """
    if channel in (START, CONVERSION, NULL):
        raise ValueError(f"Cannot compute removal effect for special state {channel}")
    if channel not in M.index:
        return 0.0  # channel never appeared in data

    P_base = conversion_probability(M)
    if P_base == 0:
        return 0.0  # no conversions to begin with — removal effect undefined

    # Build M' by zeroing the channel's row and routing everything to NULL.
    M_prime = M.copy()
    M_prime.loc[channel, :] = 0.0
    M_prime.loc[channel, NULL] = 1.0

    P_without = conversion_probability(M_prime)
    return (P_base - P_without) / P_base

# ============================================================
# 5. MARKOV ATTRIBUTION  (DataFrame in -> DataFrame out)
# ============================================================

def markov_attribution(journeys_df: pd.DataFrame) -> pd.DataFrame:
    """
    End-to-end Markov chain attribution.

    Returns a DataFrame with columns [channel, attributed_revenue, share]
    matching the format of last_click / first_click / linear / time_decay /
    shapley in models.py — so all six models can be compared in one table.

    Pipeline:
        1. build_paths(journeys_df)             — paths with START/CONV/NULL
        2. build_transition_matrix(paths)       — P(to | from)
        3. removal_effect(M, channel) per channel
        4. Normalise removal effects → channel weights summing to 1.0
        5. Multiply by total converting revenue → attributed_revenue

    Why normalise step 4:
        Raw removal effects sum to >100% because channels overlap —
        removing channel A breaks journeys that also include channel B.
        Both claim that lost conversion. Normalisation expresses each
        channel's removal effect as a share of total removal effects,
        making it comparable to other attribution models.

    Parameters
    ----------
    journeys_df : DataFrame with user_id, touch_index, channel,
                  is_converting_touch, revenue.

    Returns
    -------
    DataFrame with columns:
        - channel:           channel name
        - attributed_revenue: monetary attribution (€)
        - share:              fraction of total revenue (sums to 1.0)
        - removal_effect:     raw Markov removal effect.
    Sorted descending by attributed_revenue.
    """
    # Total converting revenue (the pie we're slicing).
    total_revenue = journeys_df.loc[
        journeys_df["is_converting_touch"] == True, "revenue"
    ].sum()

    # Build paths and transition matrix.
    paths = build_paths(journeys_df)
    M = build_transition_matrix(paths)

    # Compute removal effect for each channel (excludes special states).
    channels = [s for s in M.index if s not in (START, CONVERSION, NULL)]
    removal_effects = {ch: removal_effect(M, ch) for ch in channels}

    # Normalise into shares.
    total_re = sum(removal_effects.values())
    if total_re == 0:
        shares = {ch: 0.0 for ch in channels}
    else:
        shares = {ch: re_val / total_re for ch, re_val in removal_effects.items()}

    # Build output DataFrame with same schema as other models.
    result = pd.DataFrame({
        "channel": channels,
        "attributed_revenue": [shares[ch] * total_revenue for ch in channels],
        "share": [shares[ch] for ch in channels],
        "removal_effect": [removal_effects[ch] for ch in channels],
        "model": "markov",
    })
    return result.sort_values("attributed_revenue", ascending=False).reset_index(drop=True)