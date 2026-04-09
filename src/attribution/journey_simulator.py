"""
Journey Simulator — generates realistic multi-touch user journeys.

What this module does:
- Simulates users who see ads across multiple platforms before converting
- Models cookie rejection behavior (30-40% of users)
- Simulates different tracking methods: cookies, UTM, fingerprint, referrer
- Shows how much data GA4 loses vs cookieless tracking

Why this matters:
A user might see your ad on TikTok, search on Google, click a Meta
retargeting ad, and finally buy through a direct visit. Last-click
attribution gives 100% credit to "direct" — completely wrong.

Real-world tracking challenges we simulate:
1. Cookie rejection — 35% of users decline cookies
2. Cross-device — user sees ad on phone, buys on laptop
3. Ad blockers — ~25% of desktop users block tracking scripts
4. Safari ITP — kills 3rd-party cookies after 7 days
5. Private browsing — no cookies at all

Our approach recovers lost data through:
- UTM parameters (always work, no cookies needed)
- Lightweight fingerprinting (IP + browser + screen)
- Server-side tracking (bypasses ad blockers)
- Referrer headers (browser sends origin automatically)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

# ============================================================
# CONFIG
# ============================================================

OUTPUT_PATH = Path("data/processed/user_journeys.csv")

# Traffic channels with realistic weights
# weight = how likely a user encounters this channel
CHANNELS = {
    "tiktok_paid": {
        "weight": 0.20,
        "avg_cpc": 0.60,
        "position": "top_funnel",      # awareness — first touch
        "trackable_without_cookies": True,  # UTM works
    },
    "meta_facebook_feed": {
        "weight": 0.18,
        "avg_cpc": 1.20,
        "position": "top_funnel",
        "trackable_without_cookies": True,
    },
    "meta_instagram_stories": {
        "weight": 0.12,
        "avg_cpc": 0.90,
        "position": "top_funnel",
        "trackable_without_cookies": True,
    },
    "google_search_nonbrand": {
        "weight": 0.15,
        "avg_cpc": 3.50,
        "position": "mid_funnel",       # consideration
        "trackable_without_cookies": True,
    },
    "google_display_retargeting": {
        "weight": 0.10,
        "avg_cpc": 0.80,
        "position": "mid_funnel",
        "trackable_without_cookies": False,  # needs cookies for retargeting
    },
    "google_search_brand": {
        "weight": 0.10,
        "avg_cpc": 1.50,
        "position": "bottom_funnel",    # purchase intent
        "trackable_without_cookies": True,
    },
    "organic_search": {
        "weight": 0.08,
        "avg_cpc": 0,
        "position": "mid_funnel",
        "trackable_without_cookies": True,  # referrer header
    },
    "direct": {
        "weight": 0.07,
        "avg_cpc": 0,
        "position": "bottom_funnel",
        "trackable_without_cookies": False,  # no referrer, no UTM
    },
}

# Tracking method reliability
# Each method has a detection rate (how often it successfully identifies the user)
TRACKING_METHODS = {
    "cookie": {
        "detection_rate": 0.95,  # very reliable when accepted
        "blocked_by_rejection": True,
        "blocked_by_adblocker": False,
    },
    "utm": {
        "detection_rate": 0.90,  # works if marketer sets up UTMs correctly
        "blocked_by_rejection": False,  # UTMs are in the URL, not browser storage
        "blocked_by_adblocker": False,
    },
    "fingerprint": {
        "detection_rate": 0.70,  # IP + browser combo, not perfect
        "blocked_by_rejection": False,
        "blocked_by_adblocker": False,
    },
    "referrer": {
        "detection_rate": 0.60,  # browser sends it, but not always
        "blocked_by_rejection": False,
        "blocked_by_adblocker": True,  # some blockers strip referrer
    },
    "server_side": {
        "detection_rate": 0.92,  # very reliable, runs on server
        "blocked_by_rejection": False,
        "blocked_by_adblocker": False,  # invisible to client-side blockers
    },
}


# ============================================================
# USER JOURNEY GENERATION
# ============================================================

def generate_single_journey(user_id, start_date, rng):
    """
    Generate one user's journey from first touch to conversion (or bounce).

    A realistic journey has 1-7 touchpoints over 1-30 days.
    The funnel works: awareness → consideration → purchase.

    Parameters:
        user_id: unique user identifier
        start_date: when the journey begins
        rng: numpy random generator for reproducibility

    Returns:
        list of touchpoint dicts
    """
    # User characteristics
    cookies_accepted = rng.random() > 0.35  # 35% reject cookies
    has_adblocker = rng.random() > 0.75     # 25% have ad blockers
    converts = rng.random() < 0.12          # 12% conversion rate overall
    cross_device = rng.random() < 0.30      # 30% switch devices

    # Journey length: 1-7 touchpoints
    # Converters tend to have more touchpoints
    if converts:
        n_touches = rng.integers(2, 8)  # 2-7 touches
    else:
        n_touches = rng.integers(1, 5)  # 1-4 touches

    # Build the journey following the funnel
    channel_names = list(CHANNELS.keys())
    channel_weights = np.array([CHANNELS[c]["weight"] for c in channel_names])
    channel_weights = channel_weights / channel_weights.sum()

    touchpoints = []
    current_date = start_date + timedelta(days=int(rng.integers(0, 7)))

    for touch_idx in range(n_touches):
        # Select channel (weighted random)
        # Later touchpoints bias toward bottom-funnel channels
        adjusted_weights = channel_weights.copy()
        if touch_idx >= n_touches - 2:  # last 2 touches
            for i, name in enumerate(channel_names):
                if CHANNELS[name]["position"] == "bottom_funnel":
                    adjusted_weights[i] *= 3.0
                elif CHANNELS[name]["position"] == "mid_funnel":
                    adjusted_weights[i] *= 1.5
        adjusted_weights = adjusted_weights / adjusted_weights.sum()

        channel = rng.choice(channel_names, p=adjusted_weights)
        channel_info = CHANNELS[channel]

        # Determine which tracking methods detect this touchpoint
        tracking = {}
        for method_name, method_info in TRACKING_METHODS.items():
            is_detected = rng.random() < method_info["detection_rate"]

            # Cookie tracking fails if user rejected cookies
            if method_name == "cookie" and not cookies_accepted:
                is_detected = False

            # Some methods fail with ad blockers
            if method_info["blocked_by_adblocker"] and has_adblocker:
                is_detected = False

            # Direct traffic has no UTM or referrer
            if channel == "direct" and method_name in ["utm", "referrer"]:
                is_detected = False

            # Retargeting needs cookies
            if channel == "google_display_retargeting" and not cookies_accepted:
                # Can't retarget without cookies — skip this touchpoint
                if touch_idx > 0:
                    continue

            tracking[method_name] = is_detected

        # Is this touchpoint visible to GA4?
        # GA4 primarily relies on cookies + some referrer
        ga4_visible = tracking.get("cookie", False) or tracking.get("referrer", False)

        # Is this touchpoint visible to our enhanced tracking?
        # We use UTM + fingerprint + server-side + cookies
        enhanced_visible = any([
            tracking.get("cookie", False),
            tracking.get("utm", False),
            tracking.get("fingerprint", False),
            tracking.get("server_side", False),
        ])

        # Cost of this touchpoint
        cost = channel_info["avg_cpc"] if channel_info["avg_cpc"] > 0 else 0

        touchpoints.append({
            "user_id": user_id,
            "touch_index": touch_idx,
            "channel": channel,
            "funnel_position": channel_info["position"],
            "date": current_date.strftime("%Y-%m-%d"),
            "cost": round(cost * (0.8 + rng.random() * 0.4), 2),  # ±20% variance
            "cookies_accepted": cookies_accepted,
            "has_adblocker": has_adblocker,
            "cross_device": cross_device and touch_idx >= n_touches // 2,
            "ga4_visible": ga4_visible,
            "enhanced_visible": enhanced_visible,
            "is_converting_touch": converts and touch_idx == n_touches - 1,
            "tracking_cookie": tracking.get("cookie", False),
            "tracking_utm": tracking.get("utm", False),
            "tracking_fingerprint": tracking.get("fingerprint", False),
            "tracking_referrer": tracking.get("referrer", False),
            "tracking_server_side": tracking.get("server_side", False),
        })

        # Next touchpoint is 0-5 days later
        current_date += timedelta(days=int(rng.integers(0, 6)))

    # Add revenue for converting journeys
    if converts and touchpoints:
        revenue = round(float(rng.normal(75, 25)), 2)
        revenue = max(15, revenue)
        touchpoints[-1]["revenue"] = revenue
    else:
        for tp in touchpoints:
            tp["revenue"] = 0

    return touchpoints


def generate_journeys(n_users=2000, start_date=None, seed=42):
    """
    Generate journeys for many users.

    Parameters:
        n_users: how many users to simulate
        start_date: beginning of simulation period
        seed: random seed for reproducibility

    Returns:
        DataFrame with all touchpoints from all users
    """
    rng = np.random.default_rng(seed)

    if start_date is None:
        start_date = datetime(2026, 3, 1)

    print(f"Generating journeys for {n_users} users...")

    all_touchpoints = []
    for user_id in range(n_users):
        journey = generate_single_journey(
            user_id=f"user_{user_id:05d}",
            start_date=start_date,
            rng=rng,
        )
        all_touchpoints.extend(journey)

    df = pd.DataFrame(all_touchpoints)

    # Summary stats
    total_users = df["user_id"].nunique()
    converting_users = df[df["is_converting_touch"]]["user_id"].nunique()
    total_touchpoints = len(df)
    ga4_visible = df["ga4_visible"].sum()
    enhanced_visible = df["enhanced_visible"].sum()

    print(f"\nJourney simulation complete:")
    print(f"  Users: {total_users}")
    print(f"  Conversions: {converting_users} ({converting_users/total_users*100:.1f}%)")
    print(f"  Total touchpoints: {total_touchpoints}")
    print(f"  Avg touchpoints per user: {total_touchpoints/total_users:.1f}")
    print(f"\n  GA4 sees: {ga4_visible}/{total_touchpoints} touchpoints ({ga4_visible/total_touchpoints*100:.1f}%)")
    print(f"  Enhanced tracking sees: {enhanced_visible}/{total_touchpoints} ({enhanced_visible/total_touchpoints*100:.1f}%)")
    print(f"  GA4 data loss: {(1 - ga4_visible/total_touchpoints)*100:.1f}%")
    print(f"  Enhanced data loss: {(1 - enhanced_visible/total_touchpoints)*100:.1f}%")

    return df


def save_journeys(df, output_path=None):
    """Save journeys to CSV."""
    if output_path is None:
        output_path = OUTPUT_PATH

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nSaved {len(df)} touchpoints to {output_path}")


# ============================================================
# MAIN
# ============================================================

def run_simulator():
    """Generate and save user journeys."""
    print("=" * 60)
    print("  USER JOURNEY SIMULATOR")
    print("=" * 60)

    df = generate_journeys(n_users=2000)

    # Show tracking method breakdown
    print(f"\n{'=' * 60}")
    print("  TRACKING METHOD EFFECTIVENESS")
    print(f"{'=' * 60}")

    methods = ["tracking_cookie", "tracking_utm", "tracking_fingerprint",
               "tracking_referrer", "tracking_server_side"]
    for method in methods:
        detected = df[method].sum()
        name = method.replace("tracking_", "")
        print(f"  {name:15s}: {detected}/{len(df)} touchpoints ({detected/len(df)*100:.1f}%)")

    # Cookie rejection impact
    print(f"\n{'=' * 60}")
    print("  COOKIE REJECTION IMPACT")
    print(f"{'=' * 60}")

    cookie_users = df.groupby("user_id")["cookies_accepted"].first()
    rejected = (~cookie_users).sum()
    print(f"  Users who rejected cookies: {rejected}/{len(cookie_users)} ({rejected/len(cookie_users)*100:.1f}%)")

    save_journeys(df)

    return df


if __name__ == "__main__":
    run_simulator()
