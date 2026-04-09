"""
TikTok Ads Connector — fetches campaign data from TikTok Ads Manager.

Currently: returns realistic mock data for development.
When ready: replace fetch_data() with real API calls.

============================================================
HOW TO CONNECT A REAL TIKTOK ADS ACCOUNT:
============================================================

Step 1: Create a TikTok for Business account
   - Go to: https://ads.tiktok.com
   - Create an advertiser account

Step 2: Get API Access
   - Apply for TikTok Marketing API access:
     https://business-api.tiktok.com/portal/docs
   - Create an app in TikTok Developer Portal
   - Get your App ID and Secret

Step 3: Get your Advertiser ID
   - Go to TikTok Ads Manager → Account Settings
   - Copy the Advertiser ID

Step 4: Install the SDK
   pip install tiktok-business-api
   # or use requests directly (TikTok API is REST-based)

Step 5: Add credentials to .env:
   TIKTOK_ACCESS_TOKEN=your_access_token
   TIKTOK_ADVERTISER_ID=your_advertiser_id

Step 6: Replace fetch_data() with real API call (see commented code below)
============================================================
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.connectors.base import BaseConnector


class TikTokAdsConnector(BaseConnector):
    """
    TikTok Ads connector.

    Generates realistic mock data that mimics TikTok Ads API output:
    - In-feed video ads (native content feel, high impressions)
    - TopView ads (premium placement, expensive, high reach)
    - Spark ads (boosted organic posts, better engagement)

    TikTok specifics vs other platforms:
    - Higher impressions, lower CTR (scroll-heavy behavior)
    - Lower CPC than Google but higher than Meta
    - Younger audience = lower average order value
    - Video-first = engagement metrics matter more
    """

    def __init__(self):
        super().__init__(source_name="tiktok_ads")

    def authenticate(self):
        """
        Authenticate with TikTok Marketing API.

        MOCK: Always succeeds.
        REAL: Would use access token to init API client.

        # --- REAL IMPLEMENTATION ---
        # import os
        # import requests
        #
        # self.access_token = os.getenv("TIKTOK_ACCESS_TOKEN")
        # self.advertiser_id = os.getenv("TIKTOK_ADVERTISER_ID")
        # self.base_url = "https://business-api.tiktok.com/open_api/v1.3"
        # self.headers = {
        #     "Access-Token": self.access_token,
        #     "Content-Type": "application/json",
        # }
        """
        print(f"[{self.source_name}] Authenticated (mock mode)")
        self.is_authenticated = True

    def fetch_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch campaign data from TikTok Ads.

        MOCK: Generates realistic data for 3 campaign types.
        REAL: Would call TikTok Reporting API.

        # --- REAL IMPLEMENTATION ---
        # import requests
        #
        # url = f"{self.base_url}/report/integrated/get/"
        # params = {
        #     "advertiser_id": self.advertiser_id,
        #     "report_type": "BASIC",
        #     "dimensions": ["campaign_id", "stat_time_day"],
        #     "data_level": "AUCTION_CAMPAIGN",
        #     "start_date": start_date,
        #     "end_date": end_date,
        #     "metrics": [
        #         "campaign_name", "impressions", "clicks",
        #         "spend", "conversion", "total_complete_payment_rate",
        #         "complete_payment_roas"
        #     ],
        #     "page_size": 1000,
        # }
        # response = requests.get(url, headers=self.headers, json=params)
        # data = response.json()["data"]["list"]
        #
        # rows = []
        # for item in data:
        #     metrics = item["metrics"]
        #     dims = item["dimensions"]
        #     rows.append({
        #         "campaign_name": metrics["campaign_name"],
        #         "c_date": dims["stat_time_day"],
        #         "impressions": int(metrics["impressions"]),
        #         "clicks": int(metrics["clicks"]),
        #         "mark_spent": float(metrics["spend"]),
        #         "leads": int(float(metrics["conversion"]) * 0.4),
        #         "orders": int(float(metrics["conversion"]) * 0.6),
        #         "revenue": float(metrics["spend"]) * float(metrics.get("complete_payment_roas", 0)),
        #     })
        # df = pd.DataFrame(rows)
        # df["source"] = self.source_name
        # return df
        """
        np.random.seed(44)
        dates = pd.date_range(start=start_date, end=end_date, freq="D")

        campaigns = {
            "tiktok_infeed_video": {
                "impressions": (12000, 3000),
                "ctr": (0.012, 0.004),
                "cpc": (0.60, 0.15),
                "conv_rate": (0.02, 0.008),
                "avg_order_value": (40, 10),
            },
            "tiktok_topview": {
                "impressions": (50000, 10000),
                "ctr": (0.008, 0.003),
                "cpc": (2.50, 0.60),
                "conv_rate": (0.015, 0.005),
                "avg_order_value": (45, 12),
            },
            "tiktok_spark_ads": {
                "impressions": (8000, 2000),
                "ctr": (0.018, 0.005),
                "cpc": (0.45, 0.12),
                "conv_rate": (0.025, 0.01),
                "avg_order_value": (35, 8),
            },
        }

        rows = []
        for date in dates:
            for camp_name, params in campaigns.items():
                impressions = max(0, int(np.random.normal(*params["impressions"])))
                ctr = max(0.001, np.random.normal(*params["ctr"]))
                clicks = max(0, int(impressions * ctr))
                cpc = max(0.05, np.random.normal(*params["cpc"]))
                spend = round(clicks * cpc, 2)
                conv_rate = max(0.001, np.random.normal(*params["conv_rate"]))
                total_conversions = max(0, int(clicks * conv_rate))
                leads = max(0, int(total_conversions * 0.4))
                orders = max(0, int(total_conversions * 0.6))
                aov = max(5, np.random.normal(*params["avg_order_value"]))
                revenue = round(orders * aov, 2)

                rows.append({
                    "source": self.source_name,
                    "campaign_name": camp_name,
                    "c_date": date.strftime("%Y-%m-%d"),
                    "impressions": impressions,
                    "clicks": clicks,
                    "leads": leads,
                    "orders": orders,
                    "mark_spent": spend,
                    "revenue": revenue,
                })

        return pd.DataFrame(rows)