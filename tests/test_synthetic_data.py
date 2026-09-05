import unittest
import numpy as np
import pandas as pd
from src.data.synthetic_data import (
    generate_synthetic_market_data,
    regime_summary,
    REGIME_NAMES,
    TRANSITION_MATRIX,
)


class TestSyntheticData(unittest.TestCase):
    def test_transition_matrix_validity(self):
        """Verify stochastic transition matrix properties."""
        self.assertEqual(TRANSITION_MATRIX.shape, (5, 5))
        row_sums = TRANSITION_MATRIX.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)
        self.assertTrue(np.all(TRANSITION_MATRIX >= 0.0))

    def test_generate_synthetic_market_data_basic(self):
        """Test standard generation shape and column presence."""
        start_date = "2020-01-01"
        end_date = "2020-12-31"
        df, regimes = generate_synthetic_market_data(
            start_date=start_date, end_date=end_date, seed=42
        )

        self.assertIsInstance(df, pd.DataFrame)
        self.assertIsInstance(regimes, np.ndarray)
        self.assertEqual(len(df), len(regimes))
        self.assertGreater(len(df), 200)

        expected_cols = [
            "Close",
            "Midcap_Close",
            "Smallcap_Close",
            "IndiaVIX",
            "Advances",
            "Declines",
            "NewHighs",
            "NewLows",
            "PctAbove50DMA",
            "FII_Equity",
            "DII_Equity",
            "USDINR",
            "Gilt10Y",
            "AAA10Y",
            "SIP_Monthly",
            "TrueRegime",
        ]
        for col in expected_cols:
            self.assertIn(col, df.columns, f"Missing required column {col}")

        self.assertTrue(np.all(df["Close"] > 0))
        self.assertTrue(np.all(df["Midcap_Close"] > 0))
        self.assertTrue(np.all(df["Smallcap_Close"] > 0))
        self.assertTrue(np.all(df["IndiaVIX"] >= 8.0))

    def test_synthetic_data_regimes_validity(self):
        """Ensure regimes are bounded in [0, 4] and cover expected labels."""
        df, regimes = generate_synthetic_market_data(
            start_date="2018-01-01", end_date="2022-12-31", seed=101
        )
        unique_regimes = set(np.unique(regimes))
        self.assertTrue(unique_regimes.issubset({0, 1, 2, 3, 4}))
        for r in unique_regimes:
            self.assertIn(r, REGIME_NAMES)

    def test_synthetic_data_reproducibility(self):
        """Same seed must produce identical outputs."""
        df1, r1 = generate_synthetic_market_data("2021-01-01", "2021-06-30", seed=99)
        df2, r2 = generate_synthetic_market_data("2021-01-01", "2021-06-30", seed=99)

        pd.testing.assert_frame_equal(df1, df2)
        np.testing.assert_array_equal(r1, r2)

    def test_regime_summary(self):
        """Verify regime summary statistics."""
        df, regimes = generate_synthetic_market_data("2015-01-01", "2020-01-01", seed=123)
        summary = regime_summary(regimes)

        self.assertIsInstance(summary, pd.DataFrame)
        self.assertEqual(len(summary), 5)
        self.assertIn("count", summary.columns)
        self.assertIn("pct", summary.columns)
        self.assertIn("mean_duration", summary.columns)
        self.assertEqual(summary["count"].sum(), len(regimes))
        self.assertTrue(np.isclose(summary["pct"].sum(), 100.0, atol=0.1))


if __name__ == "__main__":
    unittest.main()
