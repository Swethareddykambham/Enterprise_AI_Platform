# datasets/datasets.py
"""
Data Loader Module for Enterprise AI Platform.

Handles loading, validating, and preprocessing datasets from the datasets directory.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Union, Dict, Any, List
import logging
import json
import requests
from io import StringIO
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class DataLoader:
    """
    Data Loader for the Enterprise AI Platform.
    
    Handles:
    - Loading datasets from files
    - Downloading datasets from URLs
    - Data validation
    - Dataset metadata management
    - Automatic dataset creation if missing
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the DataLoader.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.base_dir = Path(__file__).parent
        self.metadata_dir = self.base_dir / 'metadata'
        self.raw_dir = self.base_dir / 'raw'
        self.processed_dir = self.base_dir / 'processed'
        
        # Create directories if they don't exist
        self._create_directories()
        
        self.dataset_metadata = {}
        self.loaded_data = {}
        
        logger.info("DataLoader initialized")
    
    def _create_directories(self):
        """Create all necessary directories."""
        directories = [
            self.base_dir,
            self.metadata_dir,
            self.raw_dir,
            self.processed_dir,
            self.base_dir / 'time_series',
            self.base_dir / 'external',
            self.base_dir / 'features'
        ]
        
        for dir_path in directories:
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Directory created: {dir_path}")
    
    def load_dataset(self, dataset_name: str, 
                     force_download: bool = False) -> pd.DataFrame:
        """
        Load a dataset by name.
        
        Args:
            dataset_name: Name of the dataset to load
            force_download: Force re-download if True
        
        Returns:
            pandas DataFrame
        """
        logger.info(f"Loading dataset: {dataset_name}")
        
        # Known datasets and their URLs (updated working URLs)
        known_datasets = {
            'telco_customer_churn': {
                'url': 'https://raw.githubusercontent.com/IBM/telco-customer-churn/master/WA_Fn-UseC_-Telco-Customer-Churn.csv',
                'backup_url': 'https://raw.githubusercontent.com/plotly/datasets/master/telco-customer-churn.csv',
                'filename': 'telco_customer_churn.csv',
                'description': 'IBM Telco Customer Churn Dataset'
            },
            'customer_reviews': {
                'url': 'https://raw.githubusercontent.com/curiousily/Deep-Learning-For-Hackers/master/data/sentiment-analysis/amazon-reviews/reviews.csv',
                'backup_url': None,
                'filename': 'customer_reviews.csv',
                'description': 'Amazon Customer Reviews Dataset'
            },
            'sales_transactions': {
                'url': None,
                'backup_url': None,
                'filename': 'sales_transactions.csv',
                'description': 'Synthetic Sales Transactions'
            },
            'iris': {
                'url': 'https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv',
                'backup_url': None,
                'filename': 'iris.csv',
                'description': 'Iris Dataset'
            },
            'titanic': {
                'url': 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv',
                'backup_url': None,
                'filename': 'titanic.csv',
                'description': 'Titanic Dataset'
            },
            'tips': {
                'url': 'https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv',
                'backup_url': None,
                'filename': 'tips.csv',
                'description': 'Tips Dataset'
            }
        }
        
        if dataset_name not in known_datasets:
            # Try to load from file directly
            file_path = self.raw_dir / f'{dataset_name}.csv'
            if file_path.exists():
                logger.info(f"Loading dataset from: {file_path}")
                df = pd.read_csv(file_path)
                self.loaded_data[dataset_name] = df
                return df
            else:
                raise ValueError(f"Dataset {dataset_name} not found in known datasets")
        
        dataset_info = known_datasets[dataset_name]
        file_path = self.raw_dir / dataset_info['filename']
        processed_path = self.processed_dir / dataset_info['filename'].replace('.csv', '_processed.csv')
        
        # Check if processed data exists
        if processed_path.exists() and not force_download:
            logger.info(f"Loading processed data from: {processed_path}")
            df = pd.read_csv(processed_path)
            self.loaded_data[dataset_name] = df
            return df
        
        # Check if raw data exists
        if file_path.exists() and not force_download:
            logger.info(f"Loading raw data from: {file_path}")
            df = pd.read_csv(file_path)
        else:
            # Download or generate dataset
            if dataset_info['url']:
                try:
                    df = self._download_dataset(dataset_info['url'])
                    df.to_csv(file_path, index=False)
                except Exception as e:
                    logger.warning(f"Failed to download from primary URL: {str(e)}")
                    # Try backup URL if available
                    if dataset_info['backup_url']:
                        try:
                            logger.info(f"Trying backup URL: {dataset_info['backup_url']}")
                            df = self._download_dataset(dataset_info['backup_url'])
                            df.to_csv(file_path, index=False)
                        except Exception as e2:
                            logger.warning(f"Failed to download from backup URL: {str(e2)}")
                            # Generate synthetic data
                            logger.info("Generating synthetic data instead...")
                            df = self._generate_synthetic_data(dataset_name)
                            df.to_csv(file_path, index=False)
                    else:
                        # Generate synthetic data
                        logger.info("Generating synthetic data instead...")
                        df = self._generate_synthetic_data(dataset_name)
                        df.to_csv(file_path, index=False)
            else:
                # Generate synthetic data
                logger.info(f"Generating synthetic data for {dataset_name}...")
                df = self._generate_synthetic_data(dataset_name)
                df.to_csv(file_path, index=False)
        
        # Process and save
        df_processed = self._process_dataset(df, dataset_name)
        df_processed.to_csv(processed_path, index=False)
        
        # Save metadata
        self._save_metadata(df_processed, dataset_name)
        
        self.loaded_data[dataset_name] = df_processed
        
        logger.info(f"Dataset {dataset_name} loaded. Shape: {df_processed.shape}")
        
        return df_processed
    
    def _download_dataset(self, url: str) -> pd.DataFrame:
        """
        Download dataset from URL.
        
        Args:
            url: URL to download from
        
        Returns:
            pandas DataFrame
        """
        logger.info(f"Downloading dataset from: {url}")
        
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Try to read as CSV
            try:
                df = pd.read_csv(StringIO(response.text))
            except:
                # Try other formats
                df = pd.read_json(StringIO(response.text))
            
            logger.info(f"Downloaded {len(df)} rows")
            return df
            
        except Exception as e:
            logger.error(f"Error downloading dataset: {str(e)}")
            raise
    
    def _generate_synthetic_data(self, dataset_name: str) -> pd.DataFrame:
        """
        Generate synthetic data for testing.
        
        Args:
            dataset_name: Name of the dataset to generate
        
        Returns:
            pandas DataFrame
        """
        np.random.seed(self.random_state)
        n_samples = 1000
        
        if dataset_name == 'telco_customer_churn':
            df = pd.DataFrame({
                'customerID': [f'CUST_{i:06d}' for i in range(n_samples)],
                'gender': np.random.choice(['Male', 'Female'], n_samples),
                'SeniorCitizen': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
                'Partner': np.random.choice(['Yes', 'No'], n_samples),
                'Dependents': np.random.choice(['Yes', 'No'], n_samples),
                'tenure': np.random.randint(1, 72, n_samples),
                'PhoneService': np.random.choice(['Yes', 'No'], n_samples, p=[0.9, 0.1]),
                'MultipleLines': np.random.choice(['Yes', 'No', 'No phone service'], n_samples, p=[0.4, 0.4, 0.2]),
                'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples, p=[0.4, 0.4, 0.2]),
                'OnlineSecurity': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.3, 0.3, 0.4]),
                'OnlineBackup': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.3, 0.3, 0.4]),
                'DeviceProtection': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.3, 0.3, 0.4]),
                'TechSupport': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.3, 0.3, 0.4]),
                'StreamingTV': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.3, 0.3, 0.4]),
                'StreamingMovies': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.3, 0.3, 0.4]),
                'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples, p=[0.5, 0.3, 0.2]),
                'PaperlessBilling': np.random.choice(['Yes', 'No'], n_samples),
                'PaymentMethod': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], n_samples),
                'MonthlyCharges': np.random.uniform(20, 120, n_samples),
                'TotalCharges': np.random.uniform(100, 5000, n_samples),
                'Churn': np.random.choice(['Yes', 'No'], n_samples, p=[0.3, 0.7])
            })
            return df
        
        elif dataset_name == 'customer_reviews':
            positive_reviews = [
                "Excellent product! Highly recommended.",
                "Great quality and amazing service.",
                "Best purchase I've ever made!",
                "Fantastic product, exceeded expectations.",
                "Love this product, will buy again.",
                "Outstanding performance and value.",
                "Perfect condition, very satisfied.",
                "Amazing features and great price.",
                "Superb quality, highly recommend.",
                "Wonderful experience, 5 stars!"
            ]
            
            negative_reviews = [
                "Terrible product, waste of money.",
                "Poor quality, very disappointed.",
                "Not worth the price at all.",
                "Bad customer service, regret buying.",
                "Complete waste of money.",
                "Poor build quality, broke quickly.",
                "Very unhappy with this purchase.",
                "Horrible experience, avoid this product.",
                "Not as advertised, very misleading.",
                "Extremely disappointed with quality."
            ]
            
            neutral_reviews = [
                "Average product, nothing special.",
                "Okay quality for the price.",
                "Decent product, does the job.",
                "Fair price, acceptable quality.",
                "Good enough, not great.",
                "Reasonable product, works fine.",
                "Satisfactory experience overall.",
                "Average quality, gets the job done.",
                "Pretty good, not amazing.",
                "Fair value for the price."
            ]
            
            all_reviews = positive_reviews * 4 + negative_reviews * 4 + neutral_reviews * 2
            reviews = np.random.choice(all_reviews, n_samples)
            
            sentiments = []
            for review in reviews:
                if review in positive_reviews:
                    sentiments.append('positive')
                elif review in negative_reviews:
                    sentiments.append('negative')
                else:
                    sentiments.append('neutral')
            
            df = pd.DataFrame({
                'review_id': [f'REV_{i:06d}' for i in range(n_samples)],
                'review_text': reviews,
                'sentiment': sentiments,
                'rating': np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.1, 0.1, 0.2, 0.3, 0.3]),
                'timestamp': pd.date_range(start='2023-01-01', periods=n_samples, freq='H')
            })
            return df
        
        elif dataset_name == 'sales_transactions':
            dates = pd.date_range(start='2023-01-01', periods=n_samples, freq='D')
            df = pd.DataFrame({
                'transaction_id': [f'TX_{i:06d}' for i in range(n_samples)],
                'date': dates,
                'product_id': np.random.choice(['P001', 'P002', 'P003', 'P004', 'P005'], n_samples),
                'quantity': np.random.randint(1, 10, n_samples),
                'price': np.random.uniform(10, 100, n_samples),
                'total': np.random.uniform(10, 500, n_samples)
            })
            df['total'] = df['quantity'] * df['price']
            return df
        
        elif dataset_name == 'iris':
            species = ['setosa', 'versicolor', 'virginica']
            df = pd.DataFrame({
                'sepal_length': np.random.normal(5.8, 0.8, n_samples),
                'sepal_width': np.random.normal(3.0, 0.4, n_samples),
                'petal_length': np.random.normal(3.7, 1.7, n_samples),
                'petal_width': np.random.normal(1.2, 0.7, n_samples),
                'species': np.random.choice(species, n_samples)
            })
            return df
        
        else:
            # Generic dataset
            df = pd.DataFrame({
                'id': range(n_samples),
                'feature_1': np.random.randn(n_samples),
                'feature_2': np.random.randn(n_samples),
                'feature_3': np.random.randn(n_samples),
                'target': np.random.randint(0, 2, n_samples)
            })
            return df
    
    def _process_dataset(self, df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        """
        Process dataset based on its type.
        
        Args:
            df: Raw DataFrame
            dataset_name: Name of the dataset
        
        Returns:
            Processed DataFrame
        """
        logger.info(f"Processing dataset: {dataset_name}")
        
        df_processed = df.copy()
        
        # Dataset-specific processing
        if dataset_name == 'telco_customer_churn':
            # Convert TotalCharges to numeric
            if 'TotalCharges' in df_processed.columns:
                df_processed['TotalCharges'] = pd.to_numeric(df_processed['TotalCharges'], errors='coerce')
                df_processed['TotalCharges'] = df_processed['TotalCharges'].fillna(df_processed['TotalCharges'].median())
            
            # Convert Churn to binary
            if 'Churn' in df_processed.columns:
                df_processed['Churn_Binary'] = df_processed['Churn'].map({'Yes': 1, 'No': 0})
            
            # Drop customerID if exists
            if 'customerID' in df_processed.columns:
                df_processed = df_processed.drop('customerID', axis=1)
        
        elif dataset_name == 'customer_reviews':
            if 'review_text' in df_processed.columns:
                df_processed['review_text_clean'] = df_processed['review_text'].astype(str).str.lower()
                df_processed['review_length'] = df_processed['review_text'].str.len()
                df_processed['word_count'] = df_processed['review_text'].str.split().str.len()
        
        elif dataset_name == 'sales_transactions':
            if 'date' in df_processed.columns:
                df_processed['date'] = pd.to_datetime(df_processed['date'])
                df_processed['month'] = df_processed['date'].dt.month
                df_processed['day_of_week'] = df_processed['date'].dt.dayofweek
                df_processed['quarter'] = df_processed['date'].dt.quarter
        
        logger.info(f"Dataset processed. Shape: {df_processed.shape}")
        
        return df_processed
    
    def _save_metadata(self, df: pd.DataFrame, dataset_name: str):
        """
        Save dataset metadata.
        
        Args:
            df: Processed DataFrame
            dataset_name: Name of the dataset
        """
        metadata = {
            'name': dataset_name,
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'null_counts': df.isnull().sum().to_dict(),
            'sample': df.head(5).to_dict(),
            'statistics': df.describe().to_dict(),
            'loaded_at': str(pd.Timestamp.now())
        }
        
        metadata_path = self.metadata_dir / f'{dataset_name}_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4, default=str)
        
        logger.info(f"Metadata saved to: {metadata_path}")
    
    def get_dataset_info(self, dataset_name: str) -> Dict[str, Any]:
        """
        Get information about a dataset.
        
        Args:
            dataset_name: Name of the dataset
        
        Returns:
            Dataset metadata dictionary
        """
        metadata_path = self.metadata_dir / f'{dataset_name}_metadata.json'
        
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                return json.load(f)
        else:
            return {'error': f'No metadata found for {dataset_name}'}
    
    def list_datasets(self) -> List[str]:
        """
        List all available datasets.
        
        Returns:
            List of dataset names
        """
        datasets = []
        
        # Check raw directory
        if self.raw_dir.exists():
            for file in self.raw_dir.glob('*.csv'):
                datasets.append(file.stem)
        
        # Check processed directory
        if self.processed_dir.exists():
            for file in self.processed_dir.glob('*.csv'):
                if file.stem not in datasets:
                    datasets.append(file.stem)
        
        return datasets
    
    def load_all_datasets(self) -> Dict[str, pd.DataFrame]:
        """
        Load all available datasets.
        
        Returns:
            Dictionary of dataset name to DataFrame
        """
        datasets = {}
        for dataset_name in self.list_datasets():
            try:
                datasets[dataset_name] = self.load_dataset(dataset_name)
            except Exception as e:
                logger.error(f"Failed to load {dataset_name}: {str(e)}")
        
        return datasets


# Example usage
if __name__ == "__main__":
    print("=" * 60)
    print("Testing DataLoader")
    print("=" * 60)
    
    # Initialize DataLoader
    loader = DataLoader()
    print(f"✅ DataLoader initialized")
    print(f"Base directory: {loader.base_dir}")
    
    # List available datasets
    print("\nAvailable datasets:")
    for dataset in loader.list_datasets():
        print(f"  - {dataset}")
    
    # Load Telco Customer Churn dataset
    print("\nLoading Telco Customer Churn dataset...")
    try:
        df = loader.load_dataset('telco_customer_churn')
        print(f"✅ Loaded dataset shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"\nFirst 5 rows:")
        print(df.head())
    except Exception as e:
        print(f"❌ Error loading dataset: {str(e)}")
    
    # Load Iris dataset
    print("\nLoading Iris dataset...")
    try:
        df = loader.load_dataset('iris')
        print(f"✅ Loaded dataset shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"\nFirst 5 rows:")
        print(df.head())
    except Exception as e:
        print(f"❌ Error loading dataset: {str(e)}")
    
    print("\n" + "=" * 60)
    print("✅ DataLoader test complete!")
    print("=" * 60)
    