import unittest
import numpy as np
import pandas as pd
from src.data.synthetic_data import generate_synthetic_market_data
from src.data.feature_engineering import (
    engineer_regime_features,
    population_stability_index,
    RegimePersistencePosterior,
    covid_style_crash_alert,
    event_adjusted_conviction,
    cap_segmented_stress,
)


class TestFeatureEngineering(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df, cls.regimes = generate_synthetic_market_data(
            "2021-01-01", "2023-12-31", seed=42
        )

    def test_engineer_regime_features(self):
        """Test feature matrix creation and column consistency."""
        features = engineer_regime_features(self.df)

        self.assertIsInstance(features, pd.DataFrame)
        self.assertFalse(features.empty)
        self.assertIn("ret_1d", features.columns)
        self.assertIn("ret_21d", features.columns)
        self.assertIn("vol_parkinson_21", features.columns)
        self.assertIn("mcclellan_osc", features.columns)
        self.assertIn("vix_z_252", features.columns)
        self.assertFalse(features.isnull().any().any())

    def test_regime_persistence_posterior(self):
        """Test Bayesian conjugate Beta-Bernoulli update."""
        post = RegimePersistencePosterior(alpha=2.0, beta=1.0)
        self.assertEqual(post.posterior_mean(), 2.0 / 3.0)

        # Sequential update
        post.update(persisted=18, transitioned=2)
        self.assertEqual(post.alpha, 20.0)
        self.assertEqual(post.beta, 3.0)
        self.assertTrue(np.isclose(post.posterior_mean(), 20.0 / 23.0))

        # Credible interval checks
        ci_lo, ci_hi = post.credible_interval(0.95)
        self.assertTrue(0.0 < ci_lo < post.posterior_mean() < ci_hi < 1.0)

        summary = post.summary()
        self.assertIn("posterior_mean", summary)
        self.assertIn("ci_95_lo", summary)
        self.assertIn("ci_95_hi", summary)

    def test_population_stability_index(self):
        """Test PSI drift calculation."""
        rng = np.random.default_rng(42)
        baseline = rng.normal(0, 1, 1000)
        identical = rng.normal(0, 1, 1000)
        drifted = rng.normal(3, 1, 1000)

        psi_identical = population_stability_index(baseline, identical)
        psi_drifted = population_stability_index(baseline, drifted)

        self.assertLess(psi_identical, 0.15)
        self.assertGreater(psi_drifted, 0.25)

    def test_event_adjusted_conviction(self):
        """Test conviction halving within event horizon."""
        base_conviction = 0.80
        self.assertEqual(event_adjusted_conviction(base_conviction, days_to_event=2), 0.40)
        self.assertEqual(event_adjusted_conviction(base_conviction, days_to_event=10), 0.80)
        self.assertEqual(event_adjusted_conviction(base_conviction, days_to_event=-1), 0.80)

    def test_cap_segmented_stress(self):
        """Test market cap divergence alert."""
        balanced = {"Large Cap": 0.7, "Small Cap": 0.65}
        self.assertEqual(cap_segmented_stress(balanced, threshold=0.4), "ALIGNED")

        divergent = {"Large Cap": 0.85, "Small Cap": 0.30}
        self.assertEqual(
            cap_segmented_stress(divergent, threshold=0.4), "CAP_DIVERGENCE_WARNING"
        )

    def test_covid_style_crash_alert(self):
        """Test stress channel alerts generation."""
        alerts = covid_style_crash_alert(self.df)
        self.assertIsInstance(alerts, pd.Series)
        valid_categories = {"NORMAL", "MONITOR", "WARNING", "ACUTE RISK-OFF"}
        observed_categories = set(alerts.dropna().unique())
        self.assertTrue(observed_categories.issubset(valid_categories))


if __name__ == "__main__":
    unittest.main()
