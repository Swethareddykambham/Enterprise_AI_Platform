# core_pipeline/transformation_pipes.py
"""
Unified Sklearn ingestion pipelines & feature tools.

This module provides a comprehensive pipeline for data transformation,
feature engineering, and preprocessing.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Union, List, Tuple, Dict, Any
import logging
import joblib
import json
from datetime import datetime

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

class DataTransformationPipeline:
    """
    Unified data transformation pipeline for enterprise AI platform.
    
    Handles:
    - Data loading and validation
    - Feature engineering
    - Preprocessing (scaling, encoding)
    - Feature selection
    - Pipeline serialization
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the transformation pipeline.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.pipeline = None
        self.feature_names = None
        self.label_encoder = LabelEncoder()
        self.transformer = None
        self.metadata = {}
        
        logger.info("DataTransformationPipeline initialized")
    
    def load_data(self, data_path: Union[str, Path, pd.DataFrame]) -> pd.DataFrame:
        """
        Load data from various sources.
        
        Args:
            data_path: Path to data file or DataFrame
        
        Returns:
            Loaded DataFrame
        """
        if isinstance(data_path, pd.DataFrame):
            logger.info(f"Data loaded from DataFrame. Shape: {data_path.shape}")
            return data_path
        
        data_path = Path(data_path)
        
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        ext = data_path.suffix.lower()
        
        if ext == '.csv':
            df = pd.read_csv(data_path)
        elif ext in ['.xlsx', '.xls']:
            df = pd.read_excel(data_path)
        elif ext == '.parquet':
            df = pd.read_parquet(data_path)
        elif ext == '.json':
            df = pd.read_json(data_path)
        else:
            raise ValueError(f"Unsupported file format: {ext}")
        
        logger.info(f"Loaded data with shape: {df.shape}")
        return df
    
    def validate_data(self, df: pd.DataFrame, required_columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Validate DataFrame structure and contents.
        
        Args:
            df: DataFrame to validate
            required_columns: List of required columns
        
        Returns:
            Validation results dictionary
        """
        logger.info("Validating data...")
        
        validation_results = {
            'shape': df.shape,
            'columns': list(df.columns),
            'null_counts': df.isnull().sum().to_dict(),
            'null_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
            'duplicates': df.duplicated().sum(),
            'dtypes': df.dtypes.to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum() / 1024**2
        }
        
        if required_columns:
            missing_cols = set(required_columns) - set(df.columns)
            if missing_cols:
                logger.warning(f"Missing required columns: {missing_cols}")
                validation_results['missing_required_columns'] = list(missing_cols)
            else:
                logger.info("All required columns are present")
        
        logger.info(f"Data validation complete. Shape: {df.shape}")
        logger.info(f"Memory usage: {validation_results['memory_usage']:.2f} MB")
        
        # Save validation report
        report_path = Path('analytical_reports/metrics/data_validation_report.json')
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, 'w') as f:
            json.dump(validation_results, f, indent=4, default=str)
        
        return validation_results
    
    def create_preprocessing_pipeline(self, 
                                    numeric_features: List[str],
                                    categorical_features: List[str],
                                    target_feature: Optional[str] = None,
                                    scaling_method: str = 'standard') -> Pipeline:
        """
        Create a preprocessing pipeline for data transformation.
        
        Args:
            numeric_features: List of numeric feature names
            categorical_features: List of categorical feature names
            target_feature: Target column name (optional)
            scaling_method: Scaling method ('standard', 'minmax')
        
        Returns:
            sklearn Pipeline object
        """
        logger.info("Creating preprocessing pipeline...")
        
        # Numeric pipeline
        if scaling_method == 'standard':
            scaler = StandardScaler()
        else:
            scaler = MinMaxScaler()
        
        numeric_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', scaler)
        ])
        
        # Categorical pipeline
        categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        # Combine transformers
        preprocessor = ColumnTransformer([
            ('numeric', numeric_pipeline, numeric_features),
            ('categorical', categorical_pipeline, categorical_features)
        ])
        
        # Full pipeline
        self.pipeline = Pipeline([
            ('preprocessor', preprocessor)
        ])
        
        self.metadata['numeric_features'] = numeric_features
        self.metadata['categorical_features'] = categorical_features
        self.metadata['scaling_method'] = scaling_method
        
        logger.info(f"Pipeline created with {len(numeric_features)} numeric and {len(categorical_features)} categorical features")
        
        return self.pipeline
    
    def fit_transform(self, df: pd.DataFrame, target_column: Optional[str] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Fit and transform the data.
        
        Args:
            df: Input DataFrame
            target_column: Target column name (optional)
        
        Returns:
            Tuple of (transformed features, encoded targets)
        """
        logger.info("Fitting and transforming data...")
        
        # Separate features and target
        if target_column and target_column in df.columns:
            X = df.drop(columns=[target_column])
            y = df[target_column]
            y_encoded = self.label_encoder.fit_transform(y)
            self.metadata['target_classes'] = list(self.label_encoder.classes_)
            logger.info(f"Target classes: {self.label_encoder.classes_}")
        else:
            X = df
            y_encoded = None
        
        # Identify column types
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Create pipeline
        self.create_preprocessing_pipeline(numeric_features, categorical_features)
        
        # Transform
        X_transformed = self.pipeline.fit_transform(X)
        
        # Get feature names
        self.feature_names = numeric_features + list(
            self.pipeline.named_steps['preprocessor']
            .named_transformers_['categorical']
            .named_steps['onehot']
            .get_feature_names_out(categorical_features)
        )
        
        self.metadata['feature_names'] = self.feature_names
        self.metadata['n_features'] = X_transformed.shape[1]
        
        logger.info(f"Data transformed to shape: {X_transformed.shape}")
        logger.info(f"Number of features: {X_transformed.shape[1]}")
        
        return X_transformed, y_encoded
    
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transform new data using fitted pipeline.
        
        Args:
            df: Input DataFrame
        
        Returns:
            Transformed features array
        """
        if self.pipeline is None:
            raise ValueError("Pipeline not fitted. Call fit_transform first.")
        
        logger.info("Transforming new data...")
        X_transformed = self.pipeline.transform(df)
        
        return X_transformed
    
    def split_data(self, X: np.ndarray, y: Optional[np.ndarray] = None,
                   test_size: float = 0.2, val_size: float = 0.1) -> Dict[str, np.ndarray]:
        """
        Split data into train, validation, and test sets.
        
        Args:
            X: Features array
            y: Target array (optional)
            test_size: Test set proportion
            val_size: Validation set proportion
        
        Returns:
            Dictionary with train/val/test splits
        """
        logger.info("Splitting data...")
        
        if y is not None:
            # First split: train+val vs test
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, test_size=test_size, random_state=self.random_state, stratify=y
            )
            
            # Second split: train vs val
            val_ratio = val_size / (1 - test_size)
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=val_ratio, random_state=self.random_state, stratify=y_temp
            )
            
            results = {
                'X_train': X_train, 'y_train': y_train,
                'X_val': X_val, 'y_val': y_val,
                'X_test': X_test, 'y_test': y_test
            }
        else:
            # Unsupervised split
            X_train, X_temp = train_test_split(
                X, test_size=test_size + val_size, random_state=self.random_state
            )
            
            val_ratio = val_size / (test_size + val_size)
            X_val, X_test = train_test_split(
                X_temp, test_size=val_ratio, random_state=self.random_state
            )
            
            results = {
                'X_train': X_train,
                'X_val': X_val,
                'X_test': X_test
            }
        
        logger.info(f"Train: {results['X_train'].shape[0]}, Val: {results['X_val'].shape[0]}, Test: {results['X_test'].shape[0]}")
        
        return results
    
    def save_pipeline(self, path: Union[str, Path] = 'serialized_weights/') -> Path:
        """
        Save the fitted pipeline.
        
        Args:
            path: Save path
        
        Returns:
            Path to saved pipeline
        """
        if self.pipeline is None:
            raise ValueError("Pipeline not fitted. Cannot save.")
        
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = path / f'transformation_pipeline_{timestamp}.pkl'
        
        # Save pipeline
        joblib.dump(self.pipeline, filename)
        
        # Save metadata
        metadata_path = path / f'pipeline_metadata_{timestamp}.json'
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=4)
        
        logger.info(f"Pipeline saved to: {filename}")
        logger.info(f"Metadata saved to: {metadata_path}")
        
        return filename
    
    def load_pipeline(self, path: Union[str, Path]) -> Pipeline:
        """
        Load a saved pipeline.
        
        Args:
            path: Path to saved pipeline
        
        Returns:
            Loaded pipeline
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Pipeline file not found: {path}")
        
        self.pipeline = joblib.load(path)
        
        # Load metadata if exists
        metadata_path = path.parent / f'pipeline_metadata_{path.stem.split("_")[-1]}.json'
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
        
        logger.info(f"Pipeline loaded from: {path}")
        return self.pipeline

class FeatureEngineering:
    """
    Feature engineering utilities for the enterprise platform.
    """
    
    @staticmethod
    def create_polynomial_features(X: np.ndarray, degree: int = 2) -> np.ndarray:
        """
        Create polynomial features.
        
        Args:
            X: Input features
            degree: Polynomial degree
        
        Returns:
            Polynomial features array
        """
        from sklearn.preprocessing import PolynomialFeatures
        
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        return poly.fit_transform(X)
    
    @staticmethod
    def create_interaction_terms(df: pd.DataFrame, feature_pairs: List[Tuple[str, str]]) -> pd.DataFrame:
        """
        Create interaction terms between features.
        
        Args:
            df: Input DataFrame
            feature_pairs: List of feature name pairs
        
        Returns:
            DataFrame with interaction terms
        """
        df_result = df.copy()
        
        for feat1, feat2 in feature_pairs:
            if feat1 in df.columns and feat2 in df.columns:
                df_result[f'{feat1}_{feat2}_interaction'] = df[feat1] * df[feat2]
        
        return df_result
    
    @staticmethod
    def create_lag_features(df: pd.DataFrame, column: str, lags: List[int]) -> pd.DataFrame:
        """
        Create lag features for time series data.
        
        Args:
            df: Input DataFrame
            column: Column to create lags for
            lags: List of lag values
        
        Returns:
            DataFrame with lag features
        """
        df_result = df.copy()
        
        for lag in lags:
            df_result[f'{column}_lag_{lag}'] = df_result[column].shift(lag)
        
        return df_result
    
    @staticmethod
    def create_rolling_features(df: pd.DataFrame, column: str, windows: List[int], 
                                stats: List[str] = ['mean', 'std']) -> pd.DataFrame:
        """
        Create rolling window features.
        
        Args:
            df: Input DataFrame
            column: Column to create rolling features for
            windows: List of window sizes
            stats: List of statistics to compute
        
        Returns:
            DataFrame with rolling features
        """
        df_result = df.copy()
        
        for window in windows:
            for stat in stats:
                if stat == 'mean':
                    df_result[f'{column}_rolling_{window}_mean'] = df_result[column].rolling(window).mean()
                elif stat == 'std':
                    df_result[f'{column}_rolling_{window}_std'] = df_result[column].rolling(window).std()
                elif stat == 'min':
                    df_result[f'{column}_rolling_{window}_min'] = df_result[column].rolling(window).min()
                elif stat == 'max':
                    df_result[f'{column}_rolling_{window}_max'] = df_result[column].rolling(window).max()
        
        return df_result