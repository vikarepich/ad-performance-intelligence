import pandas as pd
import numpy as np
from pathlib import Path

RAW_PATH = Path("data/raw/Marketing.csv")
PROCESSED_PATH = Path("data/processed/features.csv")

def load_data():
    df = pd.read_csv(RAW_PATH)
    df.columns = df.columns.str.lower().str.strip()
    df["c_date"] = pd.to_datetime(df["c_date"])
    return df

def engineer_features(df):
    df = df.copy()

    # базовые метрики
    df["ctr"] = df["clicks"] / df["impressions"].replace(0, np.nan)
    df["cpc"] = df["mark_spent"] / df["clicks"].replace(0, np.nan)
    df["roas"] = df["revenue"] / df["mark_spent"].replace(0, np.nan)
    df["cpl"] = df["mark_spent"] / df["leads"].replace(0, np.nan)
    df["conversion_rate"] = df["orders"] / df["clicks"].replace(0, np.nan)

    # сортируем по кампании и дате
    df = df.sort_values(["campaign_name", "c_date"]).reset_index(drop=True)

    # week-over-week изменения по каждой кампании
    df["roas_prev"] = df.groupby("campaign_name")["roas"].shift(1)
    df["ctr_prev"] = df.groupby("campaign_name")["ctr"].shift(1)
    df["spend_prev"] = df.groupby("campaign_name")["mark_spent"].shift(1)

    df["roas_wow"] = (df["roas"] - df["roas_prev"]) / df["roas_prev"].replace(0, np.nan)
    df["ctr_wow"] = (df["ctr"] - df["ctr_prev"]) / df["ctr_prev"].replace(0, np.nan)
    df["spend_wow"] = (df["mark_spent"] - df["spend_prev"]) / df["spend_prev"].replace(0, np.nan)

    # rolling средние за 3 периода
    df["roas_rolling3"] = df.groupby("campaign_name")["roas"].transform(
        lambda x: x.rolling(3, min_periods=1).mean()
    )
    df["ctr_rolling3"] = df.groupby("campaign_name")["ctr"].transform(
        lambda x: x.rolling(3, min_periods=1).mean()
    )

    # аномалия: ROAS упал больше чем на 20% vs rolling среднее
    df["is_anomaly"] = (
        (df["roas"] < df["roas_rolling3"] * 0.8) |
        (df["ctr"] < df["ctr_rolling3"] * 0.8)
    ).astype(int)

    df = df.dropna(subset=["roas_wow", "ctr_wow"])

    return df

def save_features(df):
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)
    print(f"Saved {len(df)} rows to {PROCESSED_PATH}")

if __name__ == "__main__":
    print("Loading data...")
    df = load_data()
    print(f"Raw data: {df.shape}")

    print("Engineering features...")
    df = engineer_features(df)
    print(f"Features ready: {df.shape}")
    print(f"Anomalies found: {df['is_anomaly'].sum()}")
    print(df[["campaign_name", "c_date", "roas", "roas_wow", "is_anomaly"]].head(10))

save_features(df)
