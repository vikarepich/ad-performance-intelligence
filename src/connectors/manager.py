"""
Connector Manager — aggregates data from all ad platform connectors.

What this module does:
- Initializes all available connectors
- Fetches data from each one
- Merges everything into a single unified DataFrame
- Saves the combined data for the ETL pipeline

This is the single entry point for data collection:
    python -m src.connectors.manager

Instead of manually downloading CSV files, this module
pulls data from all platforms automatically.
"""

import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

from src.connectors.google_ads import GoogleAdsConnector
from src.connectors.meta_ads import MetaAdsConnector
from src.connectors.tiktok_ads import TikTokAdsConnector
from src.connectors.ga4 import GA4Connector

# ============================================================
# CONFIG
# ============================================================

RAW_OUTPUT_PATH = Path("data/raw/combined_campaigns.csv")

# All available connectors
# Add or remove connectors here as needed
CONNECTORS = [
    GoogleAdsConnector,
    MetaAdsConnector,
    TikTokAdsConnector,
    GA4Connector,
]


# ============================================================
# MANAGER
# ============================================================

def fetch_all(
    start_date: str = None,
    end_date: str = None,
    connectors: list = None,
) -> pd.DataFrame:
    """
    Fetch data from all connectors and merge into one DataFrame.

    Parameters:
        start_date: "YYYY-MM-DD" (default: 30 days ago)
        end_date: "YYYY-MM-DD" (default: today)
        connectors: list of connector classes to use (default: all)

    Returns:
        Combined DataFrame with data from all platforms
    """
    # Default date range: last 30 days
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

    if connectors is None:
        connectors = CONNECTORS

    print(f"Fetching data from {len(connectors)} connectors")
    print(f"Date range: {start_date} to {end_date}")
    print("=" * 60)

    all_data = []
    errors = []

    for ConnectorClass in connectors:
        connector = ConnectorClass()
        try:
            df = connector.fetch_and_validate(start_date, end_date)
            all_data.append(df)
            print(f"  ✓ {connector.source_name}: {len(df)} rows")
        except Exception as e:
            errors.append({"connector": connector.source_name, "error": str(e)})
            print(f"  ✗ {connector.source_name}: {e}")

    if not all_data:
        print("\nNo data fetched from any connector!")
        return pd.DataFrame()

    # Combine all DataFrames
    combined = pd.concat(all_data, ignore_index=True)
    combined = combined.sort_values(["source", "campaign_name", "c_date"]).reset_index(drop=True)

    print(f"\n{'=' * 60}")
    print(f"Total: {len(combined)} rows from {len(all_data)} sources")
    print(f"Campaigns: {combined['campaign_name'].nunique()}")
    print(f"Sources: {combined['source'].unique().tolist()}")

    if errors:
        print(f"\nErrors ({len(errors)}):")
        for err in errors:
            print(f"  - {err['connector']}: {err['error']}")

    return combined


def save_combined(df: pd.DataFrame, output_path: Path = None):
    """
    Save the combined DataFrame to CSV.

    This replaces the manual Marketing.csv with live data.
    The ETL pipeline can then process this file.
    """
    if output_path is None:
        output_path = RAW_OUTPUT_PATH

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nSaved {len(df)} rows to {output_path}")


def show_summary(df: pd.DataFrame):
    """Print a summary of the combined data."""
    print(f"\n{'=' * 60}")
    print("  DATA SUMMARY")
    print(f"{'=' * 60}")

    # Per-source breakdown
    for source in df["source"].unique():
        source_df = df[df["source"] == source]
        print(f"\n{source}:")
        print(f"  Rows: {len(source_df)}")
        print(f"  Campaigns: {source_df['campaign_name'].nunique()}")
        print(f"  Total spend: ${source_df['mark_spent'].sum():,.2f}")
        print(f"  Total revenue: ${source_df['revenue'].sum():,.2f}")
        if source_df["mark_spent"].sum() > 0:
            roas = source_df["revenue"].sum() / source_df["mark_spent"].sum()
            print(f"  Overall ROAS: {roas:.2f}")
        print(f"  Date range: {source_df['c_date'].min()} to {source_df['c_date'].max()}")


# ============================================================
# MAIN
# ============================================================

def run_manager():
    """
    Full data collection pipeline:
    1. Fetch from all connectors
    2. Show summary
    3. Save to CSV
    """
    print("=" * 60)
    print("  AD PERFORMANCE — DATA COLLECTION")
    print("=" * 60)

    # Fetch last 30 days of data
    df = fetch_all()

    if df.empty:
        print("No data to save.")
        return

    # Show summary
    show_summary(df)

    # Save
    save_combined(df)

    print(f"\n{'*' * 60}")
    print("  Data collection complete!")
    print(f"  Next step: python -m src.etl.pipeline")
    print(f"{'*' * 60}")

    return df


if __name__ == "__main__":
    run_manager()