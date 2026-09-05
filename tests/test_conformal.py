import unittest
import numpy as np
import pandas as pd
from src.calibration.conformal import (
    split_conformal_classifier,
    adaptive_prediction_sets,
    reliability_diagram_data,
    rolling_conformal_coverage,
)


class MockModel:
    """Mock model that returns valid probability distributions."""
    def __init__(self, n_classes=5):
        self.n_classes = n_classes

    def predict(self, X):
        rng = np.random.default_rng(len(X) + self.n_classes)
        raw = rng.uniform(0.1, 1.0, size=(len(X), self.n_classes))
        return raw / raw.sum(axis=1, keepdims=True)


class TestConformalCalibration(unittest.TestCase):
    def test_split_conformal_classifier(self):
        """Test split-conformal classifier prediction set generation and coverage guarantee."""
        model = MockModel(n_classes=5)
        rng = np.random.default_rng(42)

        n_cal = 200
        n_test = 50
        X_cal = rng.normal(size=(n_cal, 4))
        y_cal = rng.integers(0, 5, size=n_cal)
        X_test = rng.normal(size=(n_test, 4))

        alpha = 0.10
        pred_sets, q_hat, emp_cov = split_conformal_classifier(
            model, X_cal, y_cal, X_test, alpha=alpha
        )

        self.assertEqual(pred_sets.shape, (n_test, 5))
        self.assertTrue(0.0 <= q_hat <= 1.0)
        self.assertGreaterEqual(emp_cov, 1.0 - alpha - 1e-4)
        self.assertEqual(pred_sets.dtype, bool)

    def test_adaptive_prediction_sets(self):
        """Test APS (Adaptive Prediction Sets)."""
        model = MockModel(n_classes=5)
        rng = np.random.default_rng(123)

        n_cal, n_test = 150, 40
        X_cal = rng.normal(size=(n_cal, 4))
        y_cal = rng.integers(0, 5, size=n_cal)
        X_test = rng.normal(size=(n_test, 4))

        pred_sets, q_hat, set_sizes = adaptive_prediction_sets(
            model, X_cal, y_cal, X_test, alpha=0.15
        )

        self.assertEqual(pred_sets.shape, (n_test, 5))
        self.assertEqual(len(set_sizes), n_test)
        self.assertTrue(np.all(set_sizes >= 1))
        self.assertTrue(np.all(set_sizes <= 5))

    def test_reliability_diagram_data(self):
        """Test reliability diagram data computation."""
        probs_max = np.array([0.15, 0.35, 0.45, 0.75, 0.85, 0.95])
        correct = np.array([0, 0, 1, 1, 1, 1])

        df = reliability_diagram_data(probs_max, correct, n_bins=5)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIn("bin_mid", df.columns)
        self.assertIn("accuracy", df.columns)
        self.assertIn("confidence", df.columns)
        self.assertIn("gap", df.columns)
        self.assertEqual(len(df), 5)

    def test_rolling_conformal_coverage(self):
        """Test rolling coverage calculation."""
        scores = np.array([0.2] * 50 + [0.8] * 50)
        q_hat = 0.5
        series = rolling_conformal_coverage(scores, q_hat, window=30)
        self.assertIsInstance(series, pd.Series)
        self.assertEqual(len(series), len(scores))
        self.assertEqual(series.iloc[45], 1.0)
        self.assertLess(series.iloc[-1], 0.5)


if __name__ == "__main__":
    unittest.main()
