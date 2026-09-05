"""
Market Data Quality & Schema Validation Engine
==============================================
Validates multi-asset financial dataframes and regime sequences before
ingestion into inference models, backtesting engines, and feature pipelines.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd


REQUIRED_MARKET_COLUMNS = [
    "Close",
    "Midcap_Close",
    "Smallcap_Close",
    "IndiaVIX",
    "Advances",
    "Declines",
    "PctAbove50DMA",
    "FII_Equity",
    "DII_Equity",
    "USDINR",
]

POSITIVE_VALUE_COLUMNS = [
    "Close",
    "Midcap_Close",
    "Smallcap_Close",
    "IndiaVIX",
    "USDINR",
]


@dataclass
class ValidationReport:
    """Encapsulates results of data quality verification."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        status = "PASSED" if self.is_valid else "FAILED"
        err_msg = f"\n  - Errors ({len(self.errors)}):\n    " + "\n    ".join(self.errors) if self.errors else ""
        warn_msg = f"\n  - Warnings ({len(self.warnings)}):\n    " + "\n    ".join(self.warnings) if self.warnings else ""
        return f"ValidationReport [{status}]: {len(self.errors)} error(s), {len(self.warnings)} warning(s){err_msg}{warn_msg}"


class MarketDataValidator:
    """
    Validates market datasets for completeness, numerical validity,
    and structural integrity.
    """

    def __init__(
        self,
        required_columns: Optional[List[str]] = None,
        positive_columns: Optional[List[str]] = None,
        allowed_regimes: Optional[List[int]] = None,
    ):
        self.required_columns = required_columns or REQUIRED_MARKET_COLUMNS
        self.positive_columns = positive_columns or POSITIVE_VALUE_COLUMNS
        self.allowed_regimes = set(allowed_regimes or [0, 1, 2, 3, 4])

    def validate(
        self,
        df: pd.DataFrame,
        regimes: Optional[np.ndarray] = None,
    ) -> ValidationReport:
        """
        Execute full battery of data quality checks.

        Parameters
        ----------
        df : pd.DataFrame
            Market time series indexed by DatetimeIndex.
        regimes : Optional[np.ndarray]
            Array of ground-truth or inferred regime integers.

        Returns
        -------
        ValidationReport
        """
        errors: List[str] = []
        warnings: List[str] = []
        metrics: Dict[str, Any] = {
            "row_count": len(df),
            "column_count": len(df.columns) if isinstance(df, pd.DataFrame) else 0,
        }

        if not isinstance(df, pd.DataFrame):
            return ValidationReport(
                is_valid=False,
                errors=["Input must be a pandas DataFrame."],
                warnings=[],
                metrics=metrics,
            )

        if df.empty:
            return ValidationReport(
                is_valid=False,
                errors=["DataFrame is empty."],
                warnings=[],
                metrics=metrics,
            )

        # 1. Schema check
        missing_cols = [col for col in self.required_columns if col not in df.columns]
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")

        # 2. Datetime index check
        if not isinstance(df.index, pd.DatetimeIndex):
            warnings.append("DataFrame index is not a DatetimeIndex.")
        else:
            if not df.index.is_monotonic_increasing:
                errors.append("DatetimeIndex is not monotonically increasing.")
            if df.index.has_duplicates:
                errors.append("DatetimeIndex contains duplicate timestamps.")

        # 3. Non-finite values check
        for col in self.required_columns:
            if col in df.columns:
                num_nulls = int(df[col].isna().sum())
                if num_nulls > 0:
                    errors.append(f"Column '{col}' has {num_nulls} NaN or null values.")

                if pd.api.types.is_numeric_dtype(df[col]):
                    num_inf = int(np.isinf(df[col].to_numpy(dtype=float)).sum())
                    if num_inf > 0:
                        errors.append(f"Column '{col}' has {num_inf} infinite values.")

        # 4. Strictly positive columns
        for col in self.positive_columns:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                vals = df[col].to_numpy(dtype=float)
                non_positives = int((vals <= 0).sum())
                if non_positives > 0:
                    errors.append(f"Column '{col}' contains {non_positives} non-positive values (<= 0).")

        # 5. Regime sequence check (if supplied)
        if regimes is not None:
            if len(regimes) != len(df):
                errors.append(f"Length mismatch: regimes has {len(regimes)} items, df has {len(df)} rows.")
            else:
                observed_regimes = set(np.unique(regimes))
                invalid_regimes = observed_regimes - self.allowed_regimes
                if invalid_regimes:
                    errors.append(f"Regime sequence contains unexpected values: {invalid_regimes}")
            metrics["regimes_count"] = len(regimes)

        is_valid = len(errors) == 0
        return ValidationReport(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            metrics=metrics,
        )
