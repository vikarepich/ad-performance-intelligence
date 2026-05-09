"""
CSV Upload — auto-detects ad platform and maps columns to unified format.

What this module does:
- Takes a raw CSV file uploaded by the user
- Detects which platform it came from (Google Ads, Meta, TikTok, GA4)
  by looking at column names
- Maps platform-specific columns to our unified format
- Returns a clean DataFrame ready for the ETL pipeline

Why auto-detection?
Each platform exports data with different column names:
- Google Ads: "Impr.", "Cost", "Conv. value"
- Meta Ads: "Impressions", "Amount spent", "Results"
- TikTok: "Impression", "Cost", "Total complete payment amount"
- GA4: "Sessions", "Total revenue"

The user shouldn't need to know our internal format.
They just upload the file and the system figures it out.
"""

import pandas as pd
import numpy as np
from pathlib import Path

from src.connectors.base import UNIFIED_COLUMNS

# ============================================================
# PLATFORM DETECTION SIGNATURES
# ============================================================

# Each platform has unique column names that identify it.
# We check if these columns exist in the uploaded CSV.
# Order matters: we check from most specific to least specific.

PLATFORM_SIGNATURES = {
    "google_ads": {
        # Google Ads export has these distinctive columns
        "required": ["campaign", "impr."],
        "alt_required": ["campaign", "impressions", "cost"],
        "alt2_required": ["campaign", "clicks", "cost"],
    },
    "meta_ads": {
        "required": ["campaign name", "amount spent"],
        "alt_required": ["campaign name", "impressions", "spend"],
        "alt2_required": ["campaign name", "results"],
    },
    "tiktok_ads": {
        "required": ["campaign name", "impression", "cost"],
        "alt_required": ["campaign name", "click", "cost"],
    },
    "ga4": {
        "required": ["session campaign", "sessions"],
        "alt_required": ["source / medium", "sessions"],
        "alt2_required": ["session source", "sessions"],
    },
}

# ============================================================
# COLUMN MAPPINGS — platform columns → our unified columns
# ============================================================

# Each mapping: {our_column: [list of possible platform column names]}
# We try each name in order until one matches.

COLUMN_MAPPINGS = {
    "google_ads": {
        "campaign_name": ["campaign", "campaign name"],
        "c_date": ["day", "date", "reporting date"],
        "impressions": ["impr.", "impressions", "impr"],
        "clicks": ["clicks", "click"],
        "mark_spent": ["cost", "cost / conv.", "spend"],
        "leads": ["conversions", "conv.", "leads"],
        "orders": ["all conv.", "purchases", "orders", "transactions"],
        "revenue": ["conv. value", "conversion value", "all conv. value", "revenue", "total conv. value"],
    },
    "meta_ads": {
        "campaign_name": ["campaign name", "campaign"],
        "c_date": ["day", "date", "reporting starts"],
        "impressions": ["impressions", "impr"],
        "clicks": ["link clicks", "clicks (all)", "clicks"],
        "mark_spent": ["amount spent", "spend", "cost"],
        "leads": ["leads", "results", "messaging conversations started"],
        "orders": ["purchases", "orders", "website purchases", "offsite conversions"],
        "revenue": [
            "purchase roas",  # special: needs multiplication by spend
            "purchase conversion value",
            "conversion value",
            "website purchase roas",
            "revenue",
        ],
    },
    "tiktok_ads": {
        "campaign_name": ["campaign name", "campaign"],
        "c_date": ["date", "day", "time"],
        "impressions": ["impression", "impressions", "show cnt"],
        "clicks": ["click", "clicks", "click cnt"],
        "mark_spent": ["cost", "spend", "total cost"],
        "leads": ["conversion", "conversions", "form submission"],
        "orders": [
            "complete payment",
            "total complete payment",
            "purchases",
            "orders",
        ],
        "revenue": [
            "total complete payment amount",
            "complete payment amount",
            "total purchase value",
            "revenue",
            "value",
        ],
    },
    "ga4": {
        "campaign_name": [
            "session campaign",
            "campaign",
            "source / medium",
            "session source / medium",
        ],
        "c_date": ["date", "day", "nth day"],
        "impressions": ["sessions", "total users", "active users"],
        "clicks": ["sessions", "engaged sessions"],
        "mark_spent": [],  # GA4 doesn't track spend
        "leads": ["conversions", "key events", "event count"],
        "orders": [
            "ecommerce purchases",
            "purchases",
            "transactions",
            "purchase",
        ],
        "revenue": [
            "purchase revenue",
            "total revenue",
            "revenue",
            "ecommerce revenue",
        ],
    },
}


# ============================================================
# PLATFORM DETECTION
# ============================================================

def detect_platform(df):
    """
    Auto-detect which ad platform the CSV came from.

    How it works:
    1. Lowercase all column names from the uploaded file
    2. Check each platform's signature columns
    3. Return the platform name if signature matches

    Parameters:
        df: DataFrame from the uploaded CSV

    Returns:
        platform name (str) or "unknown"
    """
    cols_lower = [c.lower().strip() for c in df.columns]

    for platform, signatures in PLATFORM_SIGNATURES.items():
        # Try each set of required columns
        for key in ["required", "alt_required", "alt2_required"]:
            if key not in signatures:
                continue
            required = signatures[key]
            if all(any(req in col for col in cols_lower) for req in required):
                return platform

    return "unknown"


# ============================================================
# COLUMN MAPPING
# ============================================================

def find_column(df_columns_lower, possible_names):
    """
    Find the first matching column name from a list of possibilities.

    Strategy (order matters):
    1. Exact match (case-insensitive) — strongest signal.
    2. Partial match where the target name is contained in the actual
       column. One-way only: 'conv. value' matches a column called
       'all conv. value', but NOT a column called 'conv.' — to avoid
       short generic names accidentally matching specific ones.

    Parameters:
        df_columns_lower: list of (original_col, lowercase_col) tuples
        possible_names: list of possible names for this field, ordered
                        most-preferred first

    Returns:
        original column name (str) or None
    """
    # Pass 1: exact match wins
    for name in possible_names:
        name_lower = name.lower().strip()
        for original_col, lower_col in df_columns_lower:
            if name_lower == lower_col:
                return original_col

    # Pass 2: partial match — target name contained in actual column
    # (one-way to avoid 'conv.' matching 'conv. value')
    for name in possible_names:
        name_lower = name.lower().strip()
        for original_col, lower_col in df_columns_lower:
            if name_lower in lower_col:
                return original_col

    return None


def map_columns(df, platform):
    """
    Map platform-specific columns to our unified format.

    Parameters:
        df: original DataFrame from uploaded CSV
        platform: detected platform name

    Returns:
        mapped DataFrame with unified column names
    """
    if platform not in COLUMN_MAPPINGS:
        raise ValueError(f"No mapping defined for platform: {platform}")

    mapping = COLUMN_MAPPINGS[platform]

    # Build column name lookup: [(original, lowercase), ...]
    col_lookup = [(col, col.lower().strip()) for col in df.columns]

    result = pd.DataFrame(index=df.index)
    result["source"] = platform
    mapped_info = {}

    for our_col, possible_names in mapping.items():
        if not possible_names:
            # No mapping available (e.g. GA4 doesn't have spend)
            result[our_col] = 0
            mapped_info[our_col] = "default (0)"
            continue

        found_col = find_column(col_lookup, possible_names)
        if found_col:
            result[our_col] = df[found_col]
            mapped_info[our_col] = found_col
        else:
            result[our_col] = 0
            mapped_info[our_col] = "not found (0)"

    # Handle special case: Meta ROAS column (needs multiplication by spend)
    if platform == "meta_ads" and "revenue" in mapped_info:
        rev_col = mapped_info["revenue"]
        if "roas" in rev_col.lower():
            # ROAS × spend = revenue
            result["revenue"] = result["revenue"] * result["mark_spent"]

    return result, mapped_info


# ============================================================
# MAIN PROCESSING FUNCTION
# ============================================================

def process_uploaded_csv(file_or_path, manual_platform=None):
    """
    Process an uploaded CSV file: detect platform, map columns, validate.

    This is the main function called by Streamlit or API.

    Parameters:
        file_or_path: file path (str/Path) or file-like object (from Streamlit)
        manual_platform: override auto-detection with a specific platform

    Returns:
        dict with:
        - data: cleaned DataFrame in unified format
        - platform: detected platform name
        - mapping: which columns were mapped
        - warnings: list of any issues found
    """
    # Read the CSV
    if isinstance(file_or_path, (str, Path)):
        df = pd.read_csv(file_or_path)
    else:
        df = pd.read_csv(file_or_path)

    warnings = []

    # Detect platform
    if manual_platform:
        platform = manual_platform
    else:
        platform = detect_platform(df)

    if platform == "unknown":
        return {
            "data": None,
            "platform": "unknown",
            "mapping": {},
            "warnings": [
                "Could not auto-detect platform. "
                "Please select the platform manually or check that your CSV "
                "has standard column names from the ad platform export."
            ],
            "original_columns": df.columns.tolist(),
        }

    # Map columns
    mapped_df, mapped_info = map_columns(df, platform)

    # Parse dates
    if "c_date" in mapped_df.columns:
        try:
            mapped_df["c_date"] = pd.to_datetime(mapped_df["c_date"])
        except Exception:
            warnings.append("Could not parse dates. Check date format in your CSV.")

    # Convert numeric columns
    numeric_cols = ["impressions", "clicks", "leads", "orders", "mark_spent", "revenue"]
    for col in numeric_cols:
        if col in mapped_df.columns:
            # Remove currency symbols, commas, percentage signs
            if mapped_df[col].dtype == object:
                mapped_df[col] = (
                    mapped_df[col]
                    .astype(str)
                    .str.replace(r"[€$£¥,\s%]", "", regex=True)
                    .str.replace("--", "0", regex=False)
                )
            mapped_df[col] = pd.to_numeric(mapped_df[col], errors="coerce").fillna(0)

    # Add category based on platform
    platform_categories = {
        "google_ads": "search",
        "meta_ads": "social",
        "tiktok_ads": "social",
        "ga4": "analytics",
    }
    mapped_df["category"] = platform_categories.get(platform, "other")

    # Check for unmapped columns
    for our_col, source in mapped_info.items():
        if "not found" in source:
            warnings.append(f"Column '{our_col}' not found in CSV — set to 0")

    # Validate row count
    if len(mapped_df) == 0:
        warnings.append("No data rows found in the CSV file")

    return {
        "data": mapped_df,
        "platform": platform,
        "mapping": mapped_info,
        "warnings": warnings,
        "row_count": len(mapped_df),
        "original_columns": df.columns.tolist(),
    }


def save_uploaded_data(df, append=True):
    """
    Save uploaded data to combined_campaigns.csv.

    Parameters:
        df: processed DataFrame in unified format
        append: if True, add to existing data; if False, replace
    """
    output_path = Path("data/raw/combined_campaigns.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if append and output_path.exists():
        existing = pd.read_csv(output_path)
        combined = pd.concat([existing, df], ignore_index=True)

        # Remove exact duplicates
        key_cols = ["source", "campaign_name", "c_date"]
        available_keys = [c for c in key_cols if c in combined.columns]
        if available_keys:
            before = len(combined)
            combined = combined.drop_duplicates(subset=available_keys, keep="last")
            deduped = before - len(combined)
            if deduped > 0:
                print(f"Removed {deduped} duplicate rows")

        combined.to_csv(output_path, index=False)
        print(f"Appended {len(df)} rows. Total: {len(combined)} rows in {output_path}")
    else:
        df.to_csv(output_path, index=False)
        print(f"Saved {len(df)} rows to {output_path}")