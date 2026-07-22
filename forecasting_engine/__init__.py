# forecasting_engine/__init__.py
"""
Forecasting Engine Module for Enterprise AI Platform.

This module provides time series forecasting capabilities including:
- ARIMA/SARIMA models
- Prophet forecasting
- LSTM for time series
- Ensemble forecasting
- Trend and seasonality decomposition
"""

from .time_series import TimeSeriesForecastingEngine, TimeSeriesPreprocessor

__all__ = [
    'TimeSeriesForecastingEngine',
    'TimeSeriesPreprocessor'
]

__version__ = '1.0.0'