"""Data ingestion, synthesis, feature engineering, and validation module."""

from src.data.validator import MarketDataValidator, ValidationReport

__all__ = ["MarketDataValidator", "ValidationReport"]
