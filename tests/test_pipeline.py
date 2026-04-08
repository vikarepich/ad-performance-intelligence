"""
Tests for the ETL pipeline (src/etl/pipeline.py).

What's happening here:
- We test three functions: load_data(), engineer_features(), save_features()
- For engineer_features() we create a FAKE DataFrame (fixture),
  so tests don't depend on the real Marketing.csv file
- For load_data() we replace the file path using monkeypatch —
  we create a temporary CSV and "trick" the function into reading it

Key concepts:
- fixture (@pytest.fixture) — a function that prepares data for tests.
  Pytest automatically calls it and passes the result to the test as an argument.
- monkeypatch — a pytest tool for temporarily replacing variables.
  We use it to replace RAW_PATH with a path to a temporary file.
- tmp_path — a built-in pytest fixture that provides a temporary folder.
  After the test, the folder is automatically deleted.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

# Import the functions we're testing
from src.etl.pipeline import load_data, engineer_features, save_features


# ============================================================
# FIXTURE — fake data for tests
# ============================================================

@pytest.fixture
def sample_raw_df():
    """
    Creates a small DataFrame that mimics Marketing.csv.

    Why: we don't want to depend on the real file —
    tests should work for any developer, even without data.

    6 rows here: 2 campaigns x 3 dates.
    Enough to test:
    - basic metrics (ctr, cpc, roas...)
    - week-over-week (needs at least 2 rows per campaign)
    - rolling averages (needs at least 3 rows per campaign)
    - anomalies
    """
    data = {
        "id": [1, 2, 3, 4, 5, 6],
        "c_date": pd.to_datetime([
            "2024-01-01", "2024-01-08", "2024-01-15",
            "2024-01-01", "2024-01-08", "2024-01-15",
        ]),
        "campaign_name": [
            "facebook_tier1", "facebook_tier1", "facebook_tier1",
            "google_hot", "google_hot", "google_hot",
        ],
        "category": ["social", "social", "social", "search", "search", "search"],
        "campaign_id": [101, 101, 101, 202, 202, 202],
        "impressions": [1000, 2000, 1500, 3000, 2500, 2000],
        "mark_spent": [100, 200, 150, 300, 250, 200],
        "clicks": [50, 80, 60, 120, 100, 90],
        "leads": [10, 15, 12, 25, 20, 18],
        "orders": [5, 8, 6, 12, 10, 9],
        "revenue": [500, 300, 450, 1200, 1000, 400],
    }
    return pd.DataFrame(data)


# ============================================================
# TESTS FOR load_data()
# ============================================================

class TestLoadData:
    """
    Tests for load_data().

    Problem: load_data() reads a file from RAW_PATH.
    We don't want to depend on the real Marketing.csv.

    Solution: monkeypatch — we replace RAW_PATH with a temporary file.
    tmp_path — pytest creates a temp folder and deletes it after the test.
    """

    def test_load_data_returns_dataframe(self, sample_raw_df, tmp_path, monkeypatch):
        """Check that load_data() returns a DataFrame."""
        # Step 1: save fake data to a temporary CSV
        csv_path = tmp_path / "Marketing.csv"
        sample_raw_df.to_csv(csv_path, index=False)

        # Step 2: replace RAW_PATH in pipeline module with our temp file
        # monkeypatch.setattr(MODULE, "VARIABLE", NEW_VALUE)
        monkeypatch.setattr("src.etl.pipeline.RAW_PATH", csv_path)

        # Step 3: call the function and check the result type
        result = load_data()
        assert isinstance(result, pd.DataFrame), "load_data() should return a DataFrame"

    def test_load_data_columns_lowercase(self, sample_raw_df, tmp_path, monkeypatch):
        """Check that all columns are lowercase."""
        csv_path = tmp_path / "Marketing.csv"
        # Intentionally make columns UPPERCASE — load_data() should fix them
        upper_df = sample_raw_df.copy()
        upper_df.columns = upper_df.columns.str.upper()
        upper_df.to_csv(csv_path, index=False)

        monkeypatch.setattr("src.etl.pipeline.RAW_PATH", csv_path)

        result = load_data()

        for col in result.columns:
            assert col == col.lower(), f"Column '{col}' should be lowercase"

    def test_load_data_date_is_datetime(self, sample_raw_df, tmp_path, monkeypatch):
        """Check that c_date is converted to datetime."""
        csv_path = tmp_path / "Marketing.csv"
        sample_raw_df.to_csv(csv_path, index=False)

        monkeypatch.setattr("src.etl.pipeline.RAW_PATH", csv_path)

        result = load_data()

        # pd.api.types.is_datetime64_any_dtype — checks if a column is datetime
        assert pd.api.types.is_datetime64_any_dtype(result["c_date"]), \
            "Column c_date should be datetime type"


# ============================================================
# TESTS FOR engineer_features()
# ============================================================

class TestEngineerFeatures:
    """
    Tests for engineer_features().

    Here we pass the fake DataFrame directly —
    no monkeypatch needed because the function
    takes df as an argument (doesn't read files itself).
    """

    def test_basic_metrics_exist(self, sample_raw_df):
        """Check that all basic metric columns were created."""
        result = engineer_features(sample_raw_df)

        expected_columns = ["ctr", "cpc", "roas", "cpl", "conversion_rate"]
        for col in expected_columns:
            assert col in result.columns, f"Column '{col}' is missing from result"

    def test_ctr_calculation(self, sample_raw_df):
        """
        Check that CTR = clicks / impressions is calculated correctly.

        First row: clicks=50, impressions=1000 -> CTR=0.05
        We use np.isclose() instead of == because float numbers can have
        tiny precision errors (e.g., 0.050000000001).
        """
        result = engineer_features(sample_raw_df)

        # Find the first facebook_tier1 row by date
        row = result[
            (result["campaign_name"] == "facebook_tier1") &
            (result["c_date"] == "2024-01-01")
        ]

        # This row might be dropped by dropna, only check if it exists
        if len(row) > 0:
            expected_ctr = 50 / 1000  # 0.05
            assert np.isclose(row.iloc[0]["ctr"], expected_ctr), \
                f"CTR should be {expected_ctr}, got {row.iloc[0]['ctr']}"

    def test_roas_calculation(self, sample_raw_df):
        """Check that ROAS = revenue / mark_spent is calculated correctly."""
        result = engineer_features(sample_raw_df)

        row = result[
            (result["campaign_name"] == "facebook_tier1") &
            (result["c_date"] == "2024-01-01")
        ]

        if len(row) > 0:
            expected_roas = 500 / 100  # 5.0
            assert np.isclose(row.iloc[0]["roas"], expected_roas), \
                f"ROAS should be {expected_roas}, got {row.iloc[0]['roas']}"

    def test_wow_columns_exist(self, sample_raw_df):
        """Check that week-over-week columns were created."""
        result = engineer_features(sample_raw_df)

        wow_columns = ["roas_wow", "ctr_wow", "spend_wow"]
        for col in wow_columns:
            assert col in result.columns, f"Column '{col}' is missing"

    def test_rolling_columns_exist(self, sample_raw_df):
        """Check that rolling average columns were created."""
        result = engineer_features(sample_raw_df)

        rolling_columns = ["roas_rolling3", "ctr_rolling3"]
        for col in rolling_columns:
            assert col in result.columns, f"Column '{col}' is missing"

    def test_no_nan_in_wow_after_processing(self, sample_raw_df):
        """
        Check that wow columns have no NaN after processing.

        pipeline.py calls dropna(subset=["roas_wow", "ctr_wow"]),
        so the final DataFrame should have no NaN in these columns.
        """
        result = engineer_features(sample_raw_df)

        assert result["roas_wow"].isna().sum() == 0, "roas_wow contains NaN"
        assert result["ctr_wow"].isna().sum() == 0, "ctr_wow contains NaN"

    def test_is_anomaly_binary(self, sample_raw_df):
        """
        Check that is_anomaly contains only 0 and 1.

        set() converts the column to a set of unique values.
        issubset() checks that all values are from {0, 1}.
        """
        result = engineer_features(sample_raw_df)

        unique_values = set(result["is_anomaly"].unique())
        assert unique_values.issubset({0, 1}), \
            f"is_anomaly should only contain 0 and 1, got {unique_values}"

    def test_rows_dropped_after_feature_engineering(self, sample_raw_df):
        """
        Check that first rows of each campaign are dropped (dropna).

        We have 2 campaigns x 3 dates = 6 rows.
        First row of each campaign has no wow value (no previous row),
        so dropna removes 2 rows -> 4 remain.
        """
        result = engineer_features(sample_raw_df)

        assert len(result) == 4, \
            f"Expected 4 rows after dropna, got {len(result)}"

    def test_original_df_not_modified(self, sample_raw_df):
        """
        Check that the original DataFrame was not modified.

        pipeline.py does df = df.copy() at the start — this is important!
        Without copy() we would corrupt the original data.
        """
        original_columns = list(sample_raw_df.columns)
        original_len = len(sample_raw_df)

        engineer_features(sample_raw_df)

        assert list(sample_raw_df.columns) == original_columns, \
            "Original DataFrame was modified — new columns were added!"
        assert len(sample_raw_df) == original_len, \
            "Original DataFrame was modified — row count changed!"


# ============================================================
# TESTS FOR save_features()
# ============================================================

class TestSaveFeatures:
    """
    Tests for save_features().

    We use monkeypatch so the file is saved to a
    temporary folder, not the real data/processed/.
    """

    def test_save_creates_file(self, sample_raw_df, tmp_path, monkeypatch):
        """Check that save_features() creates a CSV file."""
        output_path = tmp_path / "processed" / "features.csv"
        monkeypatch.setattr("src.etl.pipeline.PROCESSED_PATH", output_path)

        save_features(sample_raw_df)

        assert output_path.exists(), "File features.csv was not created"

    def test_save_correct_row_count(self, sample_raw_df, tmp_path, monkeypatch):
        """Check that the saved file has the correct number of rows."""
        output_path = tmp_path / "processed" / "features.csv"
        monkeypatch.setattr("src.etl.pipeline.PROCESSED_PATH", output_path)

        save_features(sample_raw_df)

        saved_df = pd.read_csv(output_path)
        assert len(saved_df) == len(sample_raw_df), \
            f"Saved {len(saved_df)} rows, expected {len(sample_raw_df)}"


# ============================================================
# EDGE CASE: division by zero
# ============================================================

class TestEdgeCases:
    """
    Tests for edge cases.

    An edge case is an unusual situation that might break the code.
    For example: zeros, empty data, NaN values.
    """

    def test_zero_impressions_does_not_crash(self):
        """
        If impressions = 0, the function should not crash.

        pipeline.py does df["impressions"].replace(0, np.nan) —
        this is protection against division by zero.
        """
        df = pd.DataFrame({
            "id": [1, 2],
            "c_date": pd.to_datetime(["2024-01-01", "2024-01-08"]),
            "campaign_name": ["test_campaign", "test_campaign"],
            "category": ["social", "social"],
            "campaign_id": [1, 1],
            "impressions": [0, 1000],
            "mark_spent": [100, 100],
            "clicks": [10, 50],
            "leads": [5, 10],
            "orders": [2, 5],
            "revenue": [200, 500],
        })

        result = engineer_features(df)

        assert isinstance(result, pd.DataFrame), \
            "engineer_features() crashed when impressions=0"
