"""
GA4 Connector — fetches analytics data from Google Analytics 4.

Currently: returns realistic mock data for development.
When ready: replace fetch_data() with real API calls.

GA4 is different from ad platforms:
- It tracks WEBSITE behavior, not ad spend
- Shows which traffic sources convert best
- Helps attribute revenue to the right campaign
- Complements ad platform data (Google Ads shows spend, GA4 shows what happened after click)

============================================================
HOW TO CONNECT A REAL GA4 ACCOUNT:
============================================================

Step 1: Enable GA4 API
   - Go to Google Cloud Console: https://console.cloud.google.com
   - Enable "Google Analytics Data API"

Step 2: Create Service Account
   - In Cloud Console → IAM → Service Accounts
   - Create a new service account
   - Download the JSON key file
   - Save it as "ga4-credentials.json" in project root

Step 3: Grant Access
   - Go to your GA4 property → Admin → Property Access Management
   - Add the service account email (from the JSON key)
   - Grant "Viewer" role

Step 4: Get your GA4 Property ID
   - Go to GA4 → Admin → Property Settings
   - Copy the Property ID (numeric, e.g. 123456789)

Step 5: Install the SDK
   pip install google-analytics-data

Step 6: Add to .env:
   GA4_PROPERTY_ID=your_property_id
   GA4_CREDENTIALS_PATH=ga4-credentials.json

Step 7: Replace fetch_data() with real API call (see commented code below)
============================================================
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.connectors.base import BaseConnector


class GA4Connector(BaseConnector):
    """
    Google Analytics 4 connector.

    Generates realistic mock data that mimics GA4 API output:
    - Tracks sessions, conversions, and revenue BY traffic source
    - Maps traffic sources to our campaign names
    - Provides the "other side" of ad data (what happened on the website)
    """

    def __init__(self):
        super().__init__(source_name="ga4")

    def authenticate(self):
        """
        Authenticate with GA4 Data API.

        MOCK: Always succeeds.
        REAL: Would use service account credentials.

        # --- REAL IMPLEMENTATION ---
        # import os
        # from google.analytics.data_v1beta import BetaAnalyticsDataClient
        #
        # credentials_path = os.getenv("GA4_CREDENTIALS_PATH", "ga4-credentials.json")
        # os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
        # self.property_id = os.getenv("GA4_PROPERTY_ID")
        # self.client = BetaAnalyticsDataClient()
        """
        print(f"[{self.source_name}] Authenticated (mock mode)")
        self.is_authenticated = True

    def fetch_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch analytics data from GA4.

        MOCK: Generates realistic traffic source data.
        REAL: Would use RunReportRequest.

        # --- REAL IMPLEMENTATION ---
        # from google.analytics.data_v1beta import RunReportRequest
        # from google.analytics.data_v1beta.types import (
        #     DateRange, Dimension, Metric
        # )
        #
        # request = RunReportRequest(
        #     property=f"properties/{self.property_id}",
        #     dimensions=[
        #         Dimension(name="date"),
        #         Dimension(name="sessionSourceMedium"),
        #         Dimension(name="sessionCampaignName"),
        #     ],
        #     metrics=[
        #         Metric(name="sessions"),
        #         Metric(name="totalUsers"),
        #         Metric(name="conversions"),
        #         Metric(name="purchaseRevenue"),
        #         Metric(name="ecommercePurchases"),
        #     ],
        #     date_ranges=[DateRange(start_date=start_date, end_date=end_date)],
        # )
        # response = self.client.run_report(request)
        #
        # rows = []
        # for row in response.rows:
        #     date = row.dimension_values[0].value  # "20240101" format
        #     source_medium = row.dimension_values[1].value
        #     campaign = row.dimension_values[2].value
        #     rows.append({
        #         "campaign_name": f"ga4_{campaign}_{source_medium}".replace(" ", "_").lower(),
        #         "c_date": f"{date[:4]}-{date[4:6]}-{date[6:8]}",
        #         "impressions": int(row.metric_values[0].value) * 10,  # estimate from sessions
        #         "clicks": int(row.metric_values[0].value),  # sessions ≈ clicks
        #         "leads": int(float(row.metric_values[2].value) * 0.6),
        #         "orders": int(row.metric_values[4].value),
        #         "mark_spent": 0,  # GA4 doesn't track ad spend
        #         "revenue": float(row.metric_values[3].value),
        #     })
        # df = pd.DataFrame(rows)
        # df["source"] = self.source_name
        # return df
        """
        np.random.seed(45)
        dates = pd.date_range(start=start_date, end=end_date, freq="D")

        # GA4 tracks by source/medium, not campaign name
        # We map them to campaign-like names for consistency
        sources = {
            "ga4_google_cpc": {
                "sessions": (500, 120),
                "conv_rate": (0.035, 0.01),
                "avg_order_value": (75, 18),
                "bounce_rate": (0.40, 0.10),
            },
            "ga4_facebook_paid": {
                "sessions": (350, 80),
                "conv_rate": (0.025, 0.008),
                "avg_order_value": (55, 12),
                "bounce_rate": (0.55, 0.12),
            },
            "ga4_tiktok_paid": {
                "sessions": (200, 60),
                "conv_rate": (0.018, 0.007),
                "avg_order_value": (40, 10),
                "bounce_rate": (0.65, 0.10),
            },
            "ga4_organic_search": {
                "sessions": (800, 200),
                "conv_rate": (0.02, 0.005),
                "avg_order_value": (65, 15),
                "bounce_rate": (0.45, 0.10),
            },
            "ga4_direct": {
                "sessions": (300, 70),
                "conv_rate": (0.04, 0.01),
                "avg_order_value": (80, 20),
                "bounce_rate": (0.35, 0.08),
            },
        }

        rows = []
        for date in dates:
            for source_name, params in sources.items():
                sessions = max(0, int(np.random.normal(*params["sessions"])))
                # GA4 doesn't have impressions — estimate from sessions
                impressions = int(sessions * np.random.uniform(8, 15))
                clicks = sessions  # sessions ≈ clicks from GA4 perspective
                conv_rate = max(0.001, np.random.normal(*params["conv_rate"]))
                total_conversions = max(0, int(sessions * conv_rate))
                leads = max(0, int(total_conversions * 0.5))
                orders = max(0, int(total_conversions * 0.5))
                aov = max(10, np.random.normal(*params["avg_order_value"]))
                revenue = round(orders * aov, 2)

                # GA4 doesn't track ad spend — that comes from ad platforms
                # We set mark_spent to 0 for GA4 data
                rows.append({
                    "source": self.source_name,
                    "campaign_name": source_name,
                    "c_date": date.strftime("%Y-%m-%d"),
                    "impressions": impressions,
                    "clicks": clicks,
                    "leads": leads,
                    "orders": orders,
                    "mark_spent": 0,  # GA4 doesn't know spend
                    "revenue": revenue,
                })

        return pd.DataFrame(rows)