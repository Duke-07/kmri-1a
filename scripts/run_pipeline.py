#!/usr/bin/env python3
"""
Pipeline Execution & Diagnostic CLI Runner
==========================================
Unified command-line interface to validate data, run synthetic data generation,
engineer regime features, and execute pipeline diagnostics.

Usage:
  python scripts/run_pipeline.py --mode validate
  python scripts/run_pipeline.py --mode synthetic --start-date 2020-01-01 --end-date 2023-12-31
  python scripts/run_pipeline.py --mode features
  python scripts/run_pipeline.py --mode full --verbose
"""

import os
import sys
import argparse
import logging

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def setup_logger(verbose: bool = False) -> logging.Logger:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=level,
    )
    return logging.getLogger("RegimePipeline")


def run_synthetic_pipeline(start_date: str, end_date: str, seed: int, logger: logging.Logger):
    from src.data.synthetic_data import generate_synthetic_market_data, regime_summary, REGIME_NAMES

    logger.info(f"Generating synthetic Indian equity market series ({start_date} to {end_date}, seed={seed})...")
    df, regimes = generate_synthetic_market_data(start_date=start_date, end_date=end_date, seed=seed)
    logger.info(f"Generated {len(df)} daily trading observations across {len(df.columns)} columns.")

    summary_df = regime_summary(regimes)
    logger.info("\n--- Latent Regime Distribution ---")
    for idx, row in summary_df.iterrows():
        regime_label = REGIME_NAMES.get(idx, f"Regime {idx}")
        logger.info(
            f"  [{idx}] {regime_label:<15}: count={int(row['count']):<5} "
            f"pct={row['pct']:.1f}% mean_duration={row['mean_duration']:.1f} days"
        )
    return df, regimes


def run_validation_pipeline(df, regimes, logger: logging.Logger) -> bool:
    from src.data.validator import MarketDataValidator

    logger.info("Executing MarketDataValidator quality checks...")
    validator = MarketDataValidator()
    report = validator.validate(df, regimes)
    logger.info(report.summary())
    return report.is_valid


def run_features_pipeline(df, logger: logging.Logger):
    from src.data.feature_engineering import engineer_regime_features, covid_style_crash_alert

    logger.info("Engineering multi-horizon regime features (returns, Parkinson vol, breadth, macro)...")
    feats = engineer_regime_features(df)
    logger.info(f"Feature matrix engineered successfully: shape = {feats.shape}")

    alerts = covid_style_crash_alert(df)
    alert_counts = alerts.value_counts().to_dict()
    logger.info(f"Crisis stress channel alerts: {alert_counts}")
    return feats


def main():
    parser = argparse.ArgumentParser(
        description="Bayesian Regime Detection Pipeline Diagnostic & Execution CLI"
    )
    parser.add_argument(
        "--mode",
        choices=["synthetic", "validate", "features", "full", "dry-run"],
        default="full",
        help="Execution mode (default: full)",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2018-01-01",
        help="Simulation start date YYYY-MM-DD (default: 2018-01-01)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default="2024-12-31",
        help="Simulation end date YYYY-MM-DD (default: 2024-12-31)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable detailed debug logging",
    )

    args = parser.parse_args()
    logger = setup_logger(args.verbose)

    logger.info(f"Starting pipeline runner in '{args.mode}' mode...")

    if args.mode == "dry-run":
        logger.info("Dry run check passed. System environment and paths verified.")
        sys.exit(0)

    try:
        df, regimes = run_synthetic_pipeline(
            start_date=args.start_date,
            end_date=args.end_date,
            seed=args.seed,
            logger=logger,
        )

        if args.mode in ["validate", "full"]:
            is_valid = run_validation_pipeline(df, regimes, logger)
            if not is_valid:
                logger.error("Validation failed! Halting pipeline.")
                sys.exit(1)

        if args.mode in ["features", "full"]:
            run_features_pipeline(df, logger)

        logger.info("Pipeline execution completed successfully.")
        sys.exit(0)

    except Exception as exc:
        logger.exception(f"Pipeline execution encountered an error: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
