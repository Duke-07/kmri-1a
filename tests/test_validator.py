import unittest
import numpy as np
import pandas as pd
from src.data.synthetic_data import generate_synthetic_market_data
from src.data.validator import MarketDataValidator, ValidationReport


class TestMarketDataValidator(unittest.TestCase):
    def setUp(self):
        self.validator = MarketDataValidator()
        self.df, self.regimes = generate_synthetic_market_data(
            start_date="2022-01-01", end_date="2022-12-31", seed=42
        )

    def test_valid_data_passes(self):
        """Standard synthetic data should pass cleanly."""
        report = self.validator.validate(self.df, self.regimes)
        self.assertIsInstance(report, ValidationReport)
        self.assertTrue(report.is_valid)
        self.assertEqual(len(report.errors), 0)
        self.assertIn("PASSED", report.summary())

    def test_missing_column_detection(self):
        """Removing required column triggers validation failure."""
        broken_df = self.df.drop(columns=["IndiaVIX"])
        report = self.validator.validate(broken_df)
        self.assertFalse(report.is_valid)
        self.assertTrue(any("IndiaVIX" in err for err in report.errors))

    def test_negative_price_detection(self):
        """Negative price values must be flagged."""
        broken_df = self.df.copy()
        broken_df.iloc[10, broken_df.columns.get_loc("Close")] = -50.0
        report = self.validator.validate(broken_df)
        self.assertFalse(report.is_valid)
        self.assertTrue(any("Close" in err and "non-positive" in err for err in report.errors))

    def test_nan_detection(self):
        """NaN values in required columns must be caught."""
        broken_df = self.df.copy()
        broken_df.iloc[5, broken_df.columns.get_loc("USDINR")] = np.nan
        report = self.validator.validate(broken_df)
        self.assertFalse(report.is_valid)
        self.assertTrue(any("USDINR" in err and "NaN" in err for err in report.errors))

    def test_invalid_regimes(self):
        """Mismatched length or invalid regime numbers must fail."""
        mismatched_regimes = self.regimes[:-5]
        report = self.validator.validate(self.df, mismatched_regimes)
        self.assertFalse(report.is_valid)
        self.assertTrue(any("Length mismatch" in err for err in report.errors))

        invalid_value_regimes = self.regimes.copy()
        invalid_value_regimes[0] = 99
        report2 = self.validator.validate(self.df, invalid_value_regimes)
        self.assertFalse(report2.is_valid)
        self.assertTrue(any("unexpected values" in err for err in report2.errors))


if __name__ == "__main__":
    unittest.main()
