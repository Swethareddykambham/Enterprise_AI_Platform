# forecasting_engine/time_series.py
"""
Multi-model statistical forecasting tools for time series analysis.

This module provides comprehensive time series forecasting capabilities including:
- ARIMA/SARIMA models
- Prophet forecasting
- LSTM neural networks for time series
- Ensemble forecasting
- Trend and seasonality decomposition
- Statistical indicators and autocorrelation
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Union
import logging
import json
import pickle
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime, timedelta

# Statistical models
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf, pacf

# Prophet
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    print("Prophet not installed. Install with: pip install prophet")

# Machine Learning
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Deep Learning
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks

import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

class TimeSeriesPreprocessor:
    """
    Preprocessor for time series data.
    
    Handles:
    - Data loading and validation
    - Missing value imputation
    - Resampling and interpolation
    - Rolling statistics
    - Lag feature creation
    - Train/validation/test splitting
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the TimeSeriesPreprocessor.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.original_data = None
        self.processed_data = None
        
        logger.info("TimeSeriesPreprocessor initialized")
    
    def load_time_series(self, data: Union[str, Path, pd.DataFrame, pd.Series],
                         date_column: Optional[str] = None,
                         value_column: Optional[str] = None,
                         date_format: Optional[str] = None) -> pd.Series:
        """
        Load and prepare time series data.
        
        Args:
            data: Data source (file path, DataFrame, or Series)
            date_column: Name of date column (if DataFrame)
            value_column: Name of value column (if DataFrame)
            date_format: Format of dates (if string)
        
        Returns:
            pandas Series with datetime index
        """
        logger.info("Loading time series data...")
        
        # If data is file path
        if isinstance(data, (str, Path)):
            data = Path(data)
            if data.suffix == '.csv':
                df = pd.read_csv(data)
            elif data.suffix in ['.xlsx', '.xls']:
                df = pd.read_excel(data)
            else:
                raise ValueError(f"Unsupported file format: {data.suffix}")
        else:
            df = data
        
        # Convert to Series with datetime index
        if isinstance(df, pd.Series):
            if df.index.is_datetime_or_timedelta:
                ts = df
            else:
                ts = df.values
        elif isinstance(df, pd.DataFrame):
            if date_column and value_column:
                df[date_column] = pd.to_datetime(df[date_column], format=date_format)
                ts = df.set_index(date_column)[value_column]
            elif pd.api.types.is_datetime64_any_dtype(df.index):
                ts = df.iloc[:, 0] if len(df.columns) == 1 else df
            else:
                # Assume first column is date, second is value
                df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0], format=date_format)
                ts = df.set_index(df.columns[0]).iloc[:, 0]
        else:
            raise ValueError("Unsupported data format")
        
        # Ensure datetime index
        if not isinstance(ts.index, pd.DatetimeIndex):
            ts.index = pd.to_datetime(ts.index)
        
        # Sort index
        ts = ts.sort_index()
        
        self.original_data = ts
        
        logger.info(f"Loaded time series with {len(ts)} observations")
        logger.info(f"Date range: {ts.index[0]} to {ts.index[-1]}")
        logger.info(f"Frequency: {pd.infer_freq(ts.index)}")
        
        return ts
    
    def clean_series(self, ts: pd.Series, 
                     handle_missing: str = 'interpolate',
                     resample_freq: Optional[str] = None) -> pd.Series:
        """
        Clean and prepare time series.
        
        Args:
            ts: Input time series
            handle_missing: How to handle missing values ('interpolate', 'ffill', 'drop')
            resample_freq: Resampling frequency (e.g., 'D', 'W', 'M')
        
        Returns:
            Cleaned time series
        """
        logger.info("Cleaning time series...")
        
        # Make copy
        ts_clean = ts.copy()
        
        # Resample if specified
        if resample_freq:
            ts_clean = ts_clean.resample(resample_freq).mean()
        
        # Handle missing values
        missing_count = ts_clean.isnull().sum()
        if missing_count > 0:
            logger.info(f"Found {missing_count} missing values")
            
            if handle_missing == 'interpolate':
                ts_clean = ts_clean.interpolate(method='time')
            elif handle_missing == 'ffill':
                ts_clean = ts_clean.fillna(method='ffill')
                ts_clean = ts_clean.fillna(method='bfill')
            elif handle_missing == 'drop':
                ts_clean = ts_clean.dropna()
            else:
                logger.warning(f"Unknown missing handling method: {handle_missing}")
        
        self.processed_data = ts_clean
        
        logger.info(f"Cleaned series has {len(ts_clean)} observations")
        logger.info(f"Date range: {ts_clean.index[0]} to {ts_clean.index[-1]}")
        
        return ts_clean
    
    def create_features(self, ts: pd.Series,
                        lags: List[int] = [1, 3, 7, 14, 30],
                        rolling_windows: List[int] = [7, 14, 30],
                        rolling_stats: List[str] = ['mean', 'std', 'min', 'max']) -> pd.DataFrame:
        """
        Create features for time series forecasting.
        
        Args:
            ts: Input time series
            lags: List of lag values to create
            rolling_windows: List of rolling window sizes
            rolling_stats: List of statistics to compute
        
        Returns:
            DataFrame with features
        """
        logger.info("Creating time series features...")
        
        df = pd.DataFrame({'value': ts})
        
        # Lag features
        for lag in lags:
            df[f'lag_{lag}'] = df['value'].shift(lag)
        
        # Rolling statistics
        for window in rolling_windows:
            for stat in rolling_stats:
                if stat == 'mean':
                    df[f'rolling_{window}_mean'] = df['value'].rolling(window).mean()
                elif stat == 'std':
                    df[f'rolling_{window}_std'] = df['value'].rolling(window).std()
                elif stat == 'min':
                    df[f'rolling_{window}_min'] = df['value'].rolling(window).min()
                elif stat == 'max':
                    df[f'rolling_{window}_max'] = df['value'].rolling(window).max()
        
        # Time-based features
        df['year'] = df.index.year
        df['month'] = df.index.month
        df['day'] = df.index.day
        df['dayofweek'] = df.index.dayofweek
        df['quarter'] = df.index.quarter
        df['dayofyear'] = df.index.dayofyear
        df['weekofyear'] = df.index.isocalendar().week.astype(int) if hasattr(df.index, 'isocalendar') else 0
        
        # Drop NaN values
        df = df.dropna()
        
        logger.info(f"Created {len(df.columns)} features with {len(df)} observations")
        
        return df
    
    def stationarity_test(self, ts: pd.Series, 
                          significance_level: float = 0.05) -> Dict[str, Any]:
        """
        Perform Augmented Dickey-Fuller test for stationarity.
        
        Args:
            ts: Time series to test
            significance_level: Significance level for test
        
        Returns:
            Dictionary with test results
        """
        logger.info("Performing stationarity test...")
        
        result = adfuller(ts.dropna())
        
        test_result = {
            'adf_statistic': result[0],
            'p_value': result[1],
            'critical_values': result[4],
            'is_stationary': result[1] < significance_level,
            'significance_level': significance_level
        }
        
        logger.info(f"ADF Statistic: {test_result['adf_statistic']:.4f}")
        logger.info(f"p-value: {test_result['p_value']:.4f}")
        logger.info(f"Is stationary: {test_result['is_stationary']}")
        
        return test_result
    
    def decompose_series(self, ts: pd.Series, 
                         model: str = 'additive',
                         period: Optional[int] = None) -> Dict[str, Any]:
        """
        Decompose time series into trend, seasonal, and residual components.
        
        Args:
            ts: Input time series
            model: Decomposition model ('additive' or 'multiplicative')
            period: Seasonal period (auto-detect if None)
        
        Returns:
            Dictionary with decomposition components
        """
        logger.info("Decomposing time series...")
        
        if period is None:
            # Auto-detect period
            freq = pd.infer_freq(ts.index)
            if freq == 'D':
                period = 7  # Weekly seasonality
            elif freq == 'W':
                period = 52  # Yearly seasonality
            elif freq == 'M':
                period = 12  # Yearly seasonality
            elif freq == 'H':
                period = 24  # Daily seasonality
            else:
                period = 7  # Default
        
        decomposition = seasonal_decompose(ts.dropna(), model=model, period=period)
        
        result = {
            'trend': decomposition.trend,
            'seasonal': decomposition.seasonal,
            'residual': decomposition.resid,
            'observed': decomposition.observed,
            'period': period,
            'model': model
        }
        
        logger.info(f"Decomposition complete with period: {period}")
        
        return result
    
    def split_series(self, ts: pd.Series, 
                     train_size: float = 0.7,
                     val_size: float = 0.15) -> Dict[str, pd.Series]:
        """
        Split time series into train, validation, and test sets.
        
        Args:
            ts: Input time series
            train_size: Proportion for training
            val_size: Proportion for validation
        
        Returns:
            Dictionary with splits
        """
        logger.info("Splitting time series...")
        
        n = len(ts)
        train_end = int(n * train_size)
        val_end = int(n * (train_size + val_size))
        
        splits = {
            'train': ts[:train_end],
            'val': ts[train_end:val_end],
            'test': ts[val_end:],
            'train_indices': ts.index[:train_end],
            'val_indices': ts.index[train_end:val_end],
            'test_indices': ts.index[val_end:]
        }
        
        logger.info(f"Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}")
        
        return splits


class TimeSeriesForecastingEngine:
    """
    Time Series Forecasting Engine.
    
    Provides:
    - ARIMA/SARIMA models
    - Prophet forecasting
    - LSTM for time series
    - Ensemble forecasting
    - Model comparison and evaluation
    """
    
    def __init__(self, random_state: int = 42, model_dir: str = 'serialized_weights'):
        """
        Initialize the TimeSeriesForecastingEngine.
        
        Args:
            random_state: Random seed for reproducibility
            model_dir: Directory to save models
        """
        self.random_state = random_state
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.models = {}
        self.forecasts = {}
        self.metrics = {}
        self.preprocessor = TimeSeriesPreprocessor(random_state)
        
        # Set random seeds
        np.random.seed(random_state)
        tf.random.set_seed(random_state)
        
        logger.info("TimeSeriesForecastingEngine initialized")
    
    def fit_arima(self, ts: pd.Series, order: Tuple[int, int, int] = (1, 1, 1),
                  seasonal_order: Optional[Tuple[int, int, int, int]] = None) -> Dict[str, Any]:
        """
        Fit ARIMA/SARIMA model.
        
        Args:
            ts: Time series data
            order: ARIMA order (p, d, q)
            seasonal_order: Seasonal order for SARIMA (P, D, Q, S)
        
        Returns:
            Dictionary with fitted model
        """
        logger.info(f"Fitting ARIMA model with order: {order}")
        
        try:
            if seasonal_order:
                model = SARIMAX(
                    ts,
                    order=order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
                model_name = f'SARIMA{order}x{seasonal_order}'
            else:
                model = ARIMA(ts, order=order)
                model_name = f'ARIMA{order}'
            
            fitted_model = model.fit()
            
            self.models[model_name] = fitted_model
            
            logger.info(f"{model_name} fitted successfully")
            
            return {
                'model': fitted_model,
                'name': model_name,
                'summary': fitted_model.summary()
            }
            
        except Exception as e:
            logger.error(f"Error fitting ARIMA model: {str(e)}")
            raise
    
    def fit_prophet(self, ts: pd.Series, 
                    seasonality_mode: str = 'additive',
                    yearly_seasonality: bool = True,
                    weekly_seasonality: bool = True,
                    daily_seasonality: bool = False,
                    changepoint_prior_scale: float = 0.05) -> Dict[str, Any]:
        """
        Fit Prophet model.
        
        Args:
            ts: Time series data
            seasonality_mode: 'additive' or 'multiplicative'
            yearly_seasonality: Include yearly seasonality
            weekly_seasonality: Include weekly seasonality
            daily_seasonality: Include daily seasonality
            changepoint_prior_scale: Changepoint prior scale
        
        Returns:
            Dictionary with fitted model
        """
        logger.info("Fitting Prophet model...")
        
        if not PROPHET_AVAILABLE:
            raise ImportError("Prophet not installed. Install with: pip install prophet")
        
        # Prepare data for Prophet
        df_prophet = pd.DataFrame({
            'ds': ts.index,
            'y': ts.values
        })
        
        # Initialize Prophet
        model = Prophet(
            seasonality_mode=seasonality_mode,
            yearly_seasonality=yearly_seasonality,
            weekly_seasonality=weekly_seasonality,
            daily_seasonality=daily_seasonality,
            changepoint_prior_scale=changepoint_prior_scale
        )
        
        # Fit model
        fitted_model = model.fit(df_prophet)
        
        self.models['Prophet'] = fitted_model
        
        logger.info("Prophet model fitted successfully")
        
        return {
            'model': fitted_model,
            'name': 'Prophet',
            'data': df_prophet
        }
    
    def fit_lstm(self, ts: pd.Series, 
                 sequence_length: int = 10,
                 lstm_units: List[int] = [64, 32],
                 epochs: int = 100,
                 batch_size: int = 32) -> Dict[str, Any]:
        """
        Fit LSTM model for time series forecasting.
        
        Args:
            ts: Time series data
            sequence_length: Length of sequences
            lstm_units: List of LSTM units
            epochs: Number of epochs
            batch_size: Batch size
        
        Returns:
            Dictionary with fitted model
        """
        logger.info("Fitting LSTM model...")
        
        # Prepare data
        data = ts.values.reshape(-1, 1)
        
        # Scale data
        scaled_data = self.preprocessor.scaler.fit_transform(data)
        
        # Create sequences
        X, y = [], []
        for i in range(len(scaled_data) - sequence_length):
            X.append(scaled_data[i:i + sequence_length])
            y.append(scaled_data[i + sequence_length])
        
        X = np.array(X)
        y = np.array(y)
        
        # Split data
        train_size = int(len(X) * 0.7)
        val_size = int(len(X) * 0.15)
        
        X_train = X[:train_size]
        y_train = y[:train_size]
        X_val = X[train_size:train_size + val_size]
        y_val = y[train_size:train_size + val_size]
        X_test = X[train_size + val_size:]
        y_test = y[train_size + val_size:]
        
        # Build LSTM model
        model = keras.Sequential()
        model.add(keras.Input(shape=(sequence_length, 1)))
        
        for i, units in enumerate(lstm_units):
            return_sequences = i < len(lstm_units) - 1
            model.add(layers.LSTM(units, return_sequences=return_sequences))
            model.add(layers.Dropout(0.2))
        
        model.add(layers.Dense(1))
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        # Callbacks
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        )
        
        # Train
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping],
            verbose=1
        )
        
        self.models['LSTM'] = model
        self.models['LSTM_history'] = history
        
        logger.info("LSTM model fitted successfully")
        
        return {
            'model': model,
            'history': history,
            'name': 'LSTM',
            'data': {
                'X_train': X_train, 'y_train': y_train,
                'X_val': X_val, 'y_val': y_val,
                'X_test': X_test, 'y_test': y_test
            }
        }
    
    def forecast(self, model_name: str, steps: int = 30,
                 future_data: Optional[pd.DataFrame] = None) -> np.ndarray:
        """
        Generate forecasts using a trained model.
        
        Args:
            model_name: Name of the model
            steps: Number of steps to forecast
            future_data: Future data for Prophet (optional)
        
        Returns:
            Forecast array
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        
        if model_name == 'Prophet':
            # Prophet forecast
            future = model.make_future_dataframe(periods=steps)
            forecast = model.predict(future)
            forecast_values = forecast['yhat'].values[-steps:]
        elif model_name == 'LSTM':
            # LSTM forecast
            last_sequence = self.models['LSTM_data']['X_val'][-1:]
            forecast_values = []
            
            for _ in range(steps):
                pred = model.predict(last_sequence, verbose=0)
                forecast_values.append(pred[0, 0])
                # Update sequence
                last_sequence = np.roll(last_sequence, -1, axis=1)
                last_sequence[0, -1, 0] = pred[0, 0]
            
            forecast_values = np.array(forecast_values)
            # Inverse scale
            forecast_values = self.preprocessor.scaler.inverse_transform(
                forecast_values.reshape(-1, 1)
            ).flatten()
        else:
            # ARIMA/SARIMA forecast
            forecast = model.forecast(steps=steps)
            forecast_values = forecast.values if hasattr(forecast, 'values') else forecast
        
        self.forecasts[model_name] = forecast_values
        
        logger.info(f"Generated {steps} forecast steps using {model_name}")
        
        return forecast_values
    
    def evaluate_forecast(self, actual: np.ndarray, 
                          predicted: np.ndarray) -> Dict[str, float]:
        """
        Evaluate forecast accuracy.
        
        Args:
            actual: Actual values
            predicted: Predicted values
        
        Returns:
            Dictionary of metrics
        """
        return {
            'mse': mean_squared_error(actual, predicted),
            'mae': mean_absolute_error(actual, predicted),
            'rmse': np.sqrt(mean_squared_error(actual, predicted)),
            'r2': r2_score(actual, predicted),
            'mape': np.mean(np.abs((actual - predicted) / actual)) * 100
        }
    
    def compare_models(self, ts: pd.Series, 
                       forecast_horizon: int = 30) -> pd.DataFrame:
        """
        Compare multiple forecasting models.
        
        Args:
            ts: Time series data
            forecast_horizon: Number of steps to forecast
        
        Returns:
            DataFrame with comparison results
        """
        logger.info("Comparing forecasting models...")
        
        results = []
        
        # Split data
        train_size = int(len(ts) * 0.8)
        train = ts[:train_size]
        test = ts[train_size:]
        
        # Test ARIMA
        try:
            arima_result = self.fit_arima(train, order=(1, 1, 1))
            arima_forecast = self.forecast('ARIMA(1, 1, 1)', len(test))
            arima_metrics = self.evaluate_forecast(test.values, arima_forecast)
            results.append({
                'Model': 'ARIMA',
                'RMSE': arima_metrics['rmse'],
                'MAE': arima_metrics['mae'],
                'MAPE (%)': arima_metrics['mape'],
                'R²': arima_metrics['r2']
            })
        except Exception as e:
            logger.warning(f"ARIMA failed: {str(e)}")
        
        # Test Prophet
        if PROPHET_AVAILABLE:
            try:
                prophet_result = self.fit_prophet(train)
                prophet_forecast = self.forecast('Prophet', len(test))
                prophet_metrics = self.evaluate_forecast(test.values, prophet_forecast)
                results.append({
                    'Model': 'Prophet',
                    'RMSE': prophet_metrics['rmse'],
                    'MAE': prophet_metrics['mae'],
                    'MAPE (%)': prophet_metrics['mape'],
                    'R²': prophet_metrics['r2']
                })
            except Exception as e:
                logger.warning(f"Prophet failed: {str(e)}")
        
        # Test LSTM
        try:
            lstm_result = self.fit_lstm(train, sequence_length=10, epochs=50)
            lstm_forecast = self.forecast('LSTM', len(test))
            lstm_metrics = self.evaluate_forecast(test.values, lstm_forecast)
            results.append({
                'Model': 'LSTM',
                'RMSE': lstm_metrics['rmse'],
                'MAE': lstm_metrics['mae'],
                'MAPE (%)': lstm_metrics['mape'],
                'R²': lstm_metrics['r2']
            })
        except Exception as e:
            logger.warning(f"LSTM failed: {str(e)}")
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(results)
        comparison_df = comparison_df.sort_values('RMSE')
        
        logger.info(f"Best model: {comparison_df.iloc[0]['Model']}")
        
        return comparison_df
    
    def plot_forecast(self, ts: pd.Series, forecasts: Dict[str, np.ndarray],
                     title: str = 'Forecast Comparison',
                     save_path: Optional[Path] = None) -> None:
        """
        Plot actual vs forecast for multiple models.
        
        Args:
            ts: Actual time series
            forecasts: Dictionary of model forecasts
            title: Plot title
            save_path: Path to save the plot
        """
        fig, ax = plt.subplots(figsize=(14, 7))
        
        # Plot actual
        ax.plot(ts.index, ts.values, label='Actual', linewidth=2, color='black')
        
        # Plot forecasts
        colors = ['blue', 'green', 'red', 'orange', 'purple']
        for i, (model_name, forecast) in enumerate(forecasts.items()):
            forecast_index = ts.index[-len(forecast):]
            ax.plot(forecast_index, forecast, 
                   label=f'{model_name} Forecast', 
                   linewidth=2, 
                   color=colors[i % len(colors)],
                   linestyle='--')
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Forecast plot saved to: {save_path}")
        
        plt.show()
    
    def save_model(self, model_name: str) -> Path:
        """
        Save a trained model.
        
        Args:
            model_name: Name of the model to save
        
        Returns:
            Path to the saved model
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        save_dir = self.model_dir / model_name
        save_dir.mkdir(parents=True, exist_ok=True)
        
        if model_name == 'Prophet':
            # Prophet model save
            import pickle
            with open(save_dir / 'model.pkl', 'wb') as f:
                pickle.dump(model, f)
        elif model_name == 'LSTM':
            model.save(str(save_dir / 'model.h5'))
        else:
            # ARIMA/SARIMA save
            with open(save_dir / 'model.pkl', 'wb') as f:
                pickle.dump(model, f)
        
        # Save metrics if available
        if model_name in self.metrics:
            with open(save_dir / 'metrics.json', 'w') as f:
                json.dump(self.metrics[model_name], f, indent=4)
        
        logger.info(f"Model saved to: {save_dir}")
        return save_dir
    
    def load_model(self, model_name: str) -> Any:
        """
        Load a saved model.
        
        Args:
            model_name: Name of the model to load
        
        Returns:
            Loaded model
        """
        model_path = self.model_dir / model_name
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        if model_name == 'Prophet':
            import pickle
            with open(model_path / 'model.pkl', 'rb') as f:
                model = pickle.load(f)
        elif model_name == 'LSTM':
            model = keras.models.load_model(str(model_path / 'model.h5'))
        else:
            import pickle
            with open(model_path / 'model.pkl', 'rb') as f:
                model = pickle.load(f)
        
        self.models[model_name] = model
        
        logger.info(f"Model loaded from: {model_path}")
        return model


# Main function for testing
if __name__ == "__main__":
    print("=" * 60)
    print("Testing Time Series Forecasting Engine")
    print("=" * 60)
    
    # Generate sample time series data
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', periods=365, freq='D')
    trend = np.linspace(0, 10, 365)
    seasonality = 5 * np.sin(2 * np.pi * np.arange(365) / 30)  # Monthly seasonality
    noise = np.random.randn(365) * 2
    ts = pd.Series(trend + seasonality + noise, index=dates)
    
    print(f"Sample time series shape: {ts.shape}")
    
    # Initialize engine
    engine = TimeSeriesForecastingEngine()
    print(f"✅ Forecasting Engine initialized")
    
    # Test preprocessing
    print("\n1. Testing preprocessing...")
    preprocessor = TimeSeriesPreprocessor()
    cleaned_ts = preprocessor.clean_series(ts)
    print(f"   Cleaned series: {cleaned_ts.shape}")
    
    # Test decomposition
    print("\n2. Testing decomposition...")
    decomposition = preprocessor.decompose_series(cleaned_ts)
    print(f"   Decomposition components: {list(decomposition.keys())}")
    
    # Test stationarity
    print("\n3. Testing stationarity...")
    stationarity = preprocessor.stationarity_test(cleaned_ts)
    print(f"   Is stationary: {stationarity['is_stationary']}")
    
    # Test ARIMA
    print("\n4. Testing ARIMA model...")
    try:
        arima_result = engine.fit_arima(cleaned_ts, order=(1, 1, 1))
        print(f"   ARIMA model fitted successfully")
    except Exception as e:
        print(f"   ARIMA failed: {str(e)}")
    
    print("\n" + "=" * 60)
    print("✅ Tests completed!")
    print("=" * 60)