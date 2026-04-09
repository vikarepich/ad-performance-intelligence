"""
Base Connector — defines the unified data format for all ad platform connectors.

What this module does:
- Defines the standard column format that ALL connectors must output
- Provides a base class with common methods
- Ensures data from Google Ads, Meta, TikTok, GA4 all look the same
  before entering our ETL pipeline

Why a base class?
Without it, every connector would output data in a different format.
Our pipeline.py expects specific columns. The base class enforces
a contract: "every connector MUST return these columns."

This is called the Adapter Pattern in software engineering —
different APIs, same output format.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from abc import ABC, abstractmethod

# ============================================================
# UNIFIED FORMAT — all connectors must output these columns
# ============================================================

UNIFIED_COLUMNS = [
    "source",         # where data came from: "google_ads", "meta_ads", etc.
    "campaign_name",  # campaign identifier
    "c_date",         # date (YYYY-MM-DD)
    "impressions",    # how many times the ad was shown
    "clicks",         # how many times the ad was clicked
    "leads",          # form submissions, signups, etc.
    "orders",         # purchases, conversions
    "mark_spent",     # money spent (USD)
    "revenue",        # money earned (USD)
]


# ============================================================
# BASE CONNECTOR CLASS
# ============================================================

class BaseConnector(ABC):
    """
    Abstract base class for all ad platform connectors.

    What is ABC (Abstract Base Class)?
    It's a Python class that CANNOT be used directly.
    You must create a subclass (like GoogleAdsConnector)
    and implement all @abstractmethod functions.
    If you forget to implement one — Python raises an error.

    This guarantees every connector has authenticate() and fetch_data().
    """

    def __init__(self, source_name: str):
        """
        Initialize with the source name.

        Args:
            source_name: identifier like "google_ads", "meta_ads", etc.
        """
        self.source_name = source_name
        self.is_authenticated = False

    @abstractmethod
    def authenticate(self):
        """
        Authenticate with the ad platform API.
        Each connector implements its own auth flow.
        """
        pass

    @abstractmethod
    def fetch_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch campaign data for a date range.

        Args:
            start_date: "YYYY-MM-DD"
            end_date: "YYYY-MM-DD"

        Returns:
            DataFrame with UNIFIED_COLUMNS
        """
        pass

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate that the DataFrame has all required columns.

        This is a safety check — if a connector returns data
        with missing columns, we catch it here instead of
        getting a cryptic error later in the pipeline.

        Args:
            df: DataFrame to validate

        Returns:
            validated DataFrame with correct column order

        Raises:
            ValueError if required columns are missing
        """
        missing = [col for col in UNIFIED_COLUMNS if col not in df.columns]
        if missing:
            raise ValueError(
                f"Connector '{self.source_name}' is missing columns: {missing}. "
                f"Required: {UNIFIED_COLUMNS}"
            )

        # Ensure correct column order and types
        df = df[UNIFIED_COLUMNS].copy()
        df["c_date"] = pd.to_datetime(df["c_date"])
        df["source"] = self.source_name

        # Fill any NaN with 0 for numeric columns
        numeric_cols = ["impressions", "clicks", "leads", "orders", "mark_spent", "revenue"]
        df[numeric_cols] = df[numeric_cols].fillna(0)

        return df

    def fetch_and_validate(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch data and validate it in one step.

        This is the main method other code should call.
        It handles auth check, fetching, and validation.
        """
        if not self.is_authenticated:
            self.authenticate()

        df = self.fetch_data(start_date, end_date)
        df = self.validate(df)

        print(f"[{self.source_name}] Fetched {len(df)} rows "
              f"({start_date} to {end_date})")

        return df