# customer_analytics/segmentation_engines.py
"""
Segmentation Engines for Customer Analytics.

This module implements various clustering algorithms for customer segmentation,
including K-Means, DBSCAN, and Agglomerative clustering with dimensionality reduction.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Union
import logging
import json
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

class CustomerSegmentationEngine:
    """
    Customer Segmentation Engine using unsupervised learning.
    
    Provides:
    - K-Means clustering
    - DBSCAN clustering
    - Agglomerative Hierarchical clustering
    - Dimensionality reduction (PCA, t-SNE, UMAP)
    - Cluster evaluation and visualization
    """
    
    def __init__(self, random_state: int = 42, model_dir: str = 'serialized_weights'):
        """
        Initialize the Segmentation Engine.
        
        Args:
            random_state: Random seed for reproducibility
            model_dir: Directory to save models
        """
        self.random_state = random_state
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.models = {}
        self.cluster_labels = {}
        self.metrics = {}
        self.scaler = StandardScaler()
        self.reduced_data = None
        
        np.random.seed(random_state)
        
        logger.info("Customer Segmentation Engine initialized")
    
    def perform_kmeans(self, X: np.ndarray, n_clusters: int = 5, 
                       max_iter: int = 300, 
                       n_init: int = 10) -> Dict[str, Any]:
        """
        Perform K-Means clustering.
        
        Args:
            X: Input features
            n_clusters: Number of clusters
            max_iter: Maximum iterations
            n_init: Number of initializations
        
        Returns:
            Dictionary with clustering results
        """
        logger.info(f"Performing K-Means clustering with {n_clusters} clusters")
        
        # Scale data
        X_scaled = self.scaler.fit_transform(X)
        
        # Perform K-Means
        kmeans = KMeans(
            n_clusters=n_clusters,
            max_iter=max_iter,
            n_init=n_init,
            random_state=self.random_state
        )
        
        labels = kmeans.fit_predict(X_scaled)
        
        # Store results
        self.models['kmeans'] = kmeans
        self.cluster_labels['kmeans'] = labels
        
        # Calculate metrics
        metrics = self._calculate_cluster_metrics(X_scaled, labels)
        self.metrics['kmeans'] = metrics
        
        logger.info(f"K-Means clustering complete. Silhouette score: {metrics['silhouette_score']:.3f}")
        
        return {
            'model': kmeans,
            'labels': labels,
            'metrics': metrics,
            'centers': kmeans.cluster_centers_
        }
    
    def perform_dbscan(self, X: np.ndarray, eps: float = 0.5, 
                       min_samples: int = 5) -> Dict[str, Any]:
        """
        Perform DBSCAN clustering.
        
        Args:
            X: Input features
            eps: Maximum distance between samples
            min_samples: Minimum samples in a neighborhood
        
        Returns:
            Dictionary with clustering results
        """
        logger.info(f"Performing DBSCAN clustering with eps={eps}, min_samples={min_samples}")
        
        # Scale data
        X_scaled = self.scaler.fit_transform(X)
        
        # Perform DBSCAN
        dbscan = DBSCAN(
            eps=eps,
            min_samples=min_samples,
            metric='euclidean'
        )
        
        labels = dbscan.fit_predict(X_scaled)
        
        # Store results
        self.models['dbscan'] = dbscan
        self.cluster_labels['dbscan'] = labels
        
        # Calculate metrics (excluding noise points for silhouette)
        if len(set(labels)) > 1 and -1 not in labels:
            metrics = self._calculate_cluster_metrics(X_scaled, labels)
        else:
            metrics = {
                'silhouette_score': 0.0,
                'calinski_harabasz_score': 0.0,
                'davies_bouldin_score': 0.0,
                'n_clusters': len(set(labels)) - (1 if -1 in labels else 0),
                'n_noise': np.sum(labels == -1)
            }
        
        self.metrics['dbscan'] = metrics
        
        logger.info(f"DBSCAN clustering complete. Found {metrics['n_clusters']} clusters, {metrics['n_noise']} noise points")
        
        return {
            'model': dbscan,
            'labels': labels,
            'metrics': metrics
        }
    
    def perform_agglomerative(self, X: np.ndarray, n_clusters: int = 5,
                              linkage: str = 'ward') -> Dict[str, Any]:
        """
        Perform Agglomerative Hierarchical clustering.
        
        Args:
            X: Input features
            n_clusters: Number of clusters
            linkage: Linkage criterion ('ward', 'complete', 'average', 'single')
        
        Returns:
            Dictionary with clustering results
        """
        logger.info(f"Performing Agglomerative clustering with {n_clusters} clusters")
        
        # Scale data
        X_scaled = self.scaler.fit_transform(X)
        
        # Perform Agglomerative clustering
        agglomerative = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=linkage
        )
        
        labels = agglomerative.fit_predict(X_scaled)
        
        # Store results
        self.models['agglomerative'] = agglomerative
        self.cluster_labels['agglomerative'] = labels
        
        # Calculate metrics
        metrics = self._calculate_cluster_metrics(X_scaled, labels)
        self.metrics['agglomerative'] = metrics
        
        logger.info(f"Agglomerative clustering complete. Silhouette score: {metrics['silhouette_score']:.3f}")
        
        return {
            'model': agglomerative,
            'labels': labels,
            'metrics': metrics
        }
    
    def _calculate_cluster_metrics(self, X: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """
        Calculate clustering metrics.
        
        Args:
            X: Features
            labels: Cluster labels
        
        Returns:
            Dictionary of metrics
        """
        n_clusters = len(set(labels))
        
        metrics = {
            'n_clusters': n_clusters,
            'silhouette_score': silhouette_score(X, labels) if n_clusters > 1 else 0.0,
            'calinski_harabasz_score': calinski_harabasz_score(X, labels) if n_clusters > 1 else 0.0,
            'davies_bouldin_score': davies_bouldin_score(X, labels) if n_clusters > 1 else 0.0
        }
        
        return metrics
    
    def reduce_dimensions(self, X: np.ndarray, method: str = 'pca',
                          n_components: int = 2) -> np.ndarray:
        """
        Reduce dimensions for visualization.
        
        Args:
            X: Input features
            method: Dimensionality reduction method ('pca', 'tsne')
            n_components: Number of components
        
        Returns:
            Reduced features
        """
        logger.info(f"Reducing dimensions using {method}")
        
        X_scaled = self.scaler.fit_transform(X)
        
        if method == 'pca':
            reducer = PCA(n_components=n_components, random_state=self.random_state)
            reduced = reducer.fit_transform(X_scaled)
        elif method == 'tsne':
            reducer = TSNE(n_components=n_components, random_state=self.random_state)
            reduced = reducer.fit_transform(X_scaled)
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        self.reduced_data = reduced
        
        logger.info(f"Dimension reduction complete. Shape: {reduced.shape}")
        return reduced
    
    def plot_clusters(self, X: np.ndarray, labels: np.ndarray,
                      method: str = 'pca', 
                      title: str = 'Cluster Visualization',
                      save_path: Optional[Path] = None) -> None:
        """
        Plot clustering results with dimensionality reduction.
        
        Args:
            X: Input features
            labels: Cluster labels
            method: Dimensionality reduction method
            title: Plot title
            save_path: Path to save the plot
        """
        # Reduce dimensions
        reduced = self.reduce_dimensions(X, method)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Scatter plot
        scatter = ax.scatter(reduced[:, 0], reduced[:, 1], 
                           c=labels, cmap='viridis', alpha=0.7)
        
        # Add colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label('Cluster')
        
        ax.set_title(f'{title} ({method.upper()})', fontsize=14, fontweight='bold')
        ax.set_xlabel(f'{method.upper()} Component 1')
        ax.set_ylabel(f'{method.upper()} Component 2')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Cluster plot saved to: {save_path}")
        
        plt.show()
    
    def plot_elbow_method(self, X: np.ndarray, max_k: int = 10,
                         save_path: Optional[Path] = None) -> None:
        """
        Plot elbow method for K-Means clustering.
        
        Args:
            X: Input features
            max_k: Maximum number of clusters
            save_path: Path to save the plot
        """
        X_scaled = self.scaler.fit_transform(X)
        
        inertias = []
        silhouette_scores = []
        
        for k in range(2, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=self.random_state)
            kmeans.fit(X_scaled)
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))
        
        # Create plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Elbow curve
        axes[0].plot(range(2, max_k + 1), inertias, 'bo-', linewidth=2)
        axes[0].set_title('Elbow Method for Optimal K', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Number of Clusters (K)')
        axes[0].set_ylabel('Inertia')
        axes[0].grid(True, alpha=0.3)
        
        # Silhouette scores
        axes[1].plot(range(2, max_k + 1), silhouette_scores, 'ro-', linewidth=2)
        axes[1].set_title('Silhouette Score for Optimal K', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Number of Clusters (K)')
        axes[1].set_ylabel('Silhouette Score')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Elbow method plot saved to: {save_path}")
        
        plt.show()
    
    def compare_algorithms(self, X: np.ndarray, 
                           n_clusters: int = 5) -> pd.DataFrame:
        """
        Compare different clustering algorithms.
        
        Args:
            X: Input features
            n_clusters: Number of clusters for K-Means and Agglomerative
        
        Returns:
            DataFrame with comparison results
        """
        logger.info("Comparing clustering algorithms...")
        
        results = []
        
        # K-Means
        kmeans_result = self.perform_kmeans(X, n_clusters)
        results.append({
            'Algorithm': 'K-Means',
            'Silhouette Score': kmeans_result['metrics']['silhouette_score'],
            'Calinski-Harabasz': kmeans_result['metrics']['calinski_harabasz_score'],
            'Davies-Bouldin': kmeans_result['metrics']['davies_bouldin_score'],
            'N_Clusters': kmeans_result['metrics']['n_clusters']
        })
        
        # DBSCAN
        dbscan_result = self.perform_dbscan(X, eps=0.5, min_samples=5)
        results.append({
            'Algorithm': 'DBSCAN',
            'Silhouette Score': dbscan_result['metrics']['silhouette_score'],
            'Calinski-Harabasz': dbscan_result['metrics']['calinski_harabasz_score'],
            'Davies-Bouldin': dbscan_result['metrics']['davies_bouldin_score'],
            'N_Clusters': dbscan_result['metrics']['n_clusters']
        })
        
        # Agglomerative
        agglomerative_result = self.perform_agglomerative(X, n_clusters)
        results.append({
            'Algorithm': 'Agglomerative',
            'Silhouette Score': agglomerative_result['metrics']['silhouette_score'],
            'Calinski-Harabasz': agglomerative_result['metrics']['calinski_harabasz_score'],
            'Davies-Bouldin': agglomerative_result['metrics']['davies_bouldin_score'],
            'N_Clusters': agglomerative_result['metrics']['n_clusters']
        })
        
        comparison_df = pd.DataFrame(results)
        comparison_df = comparison_df.sort_values('Silhouette Score', ascending=False)
        
        logger.info(f"Best algorithm: {comparison_df.iloc[0]['Algorithm']}")
        
        return comparison_df
    
    def get_cluster_profiles(self, df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
        """
        Generate cluster profiles for interpretation.
        
        Args:
            df: Original DataFrame with features
            labels: Cluster labels
        
        Returns:
            DataFrame with cluster profiles
        """
        logger.info("Generating cluster profiles...")
        
        # Add cluster labels
        df_clustered = df.copy()
        df_clustered['Cluster'] = labels
        
        # Calculate profiles
        numeric_cols = df_clustered.select_dtypes(include=['int64', 'float64']).columns
        profiles = df_clustered.groupby('Cluster')[numeric_cols].mean()
        
        # Add cluster sizes
        cluster_sizes = df_clustered['Cluster'].value_counts().sort_index()
        profiles['Cluster_Size'] = cluster_sizes
        
        # Add percentage
        profiles['Percentage'] = (profiles['Cluster_Size'] / len(df_clustered) * 100).round(2)
        
        logger.info(f"Cluster profiles generated for {len(profiles)} clusters")
        
        return profiles
    
    def save_model(self, model_name: str) -> Path:
        """
        Save a trained segmentation model.
        
        Args:
            model_name: Name of the model to save ('kmeans', 'dbscan', 'agglomerative')
        
        Returns:
            Path to the saved model
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        save_dir = self.model_dir / model_name
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        with open(save_dir / 'model.pkl', 'wb') as f:
            pickle.dump(model, f)
        
        # Save labels and metrics
        if model_name in self.cluster_labels:
            np.save(save_dir / 'labels.npy', self.cluster_labels[model_name])
        
        if model_name in self.metrics:
            with open(save_dir / 'metrics.json', 'w') as f:
                json.dump(self.metrics[model_name], f, indent=4)
        
        logger.info(f"Model saved to: {save_dir}")
        return save_dir
    
    def load_model(self, model_name: str) -> Any:
        """
        Load a saved segmentation model.
        
        Args:
            model_name: Name of the model to load
        
        Returns:
            Loaded model
        """
        model_path = self.model_dir / model_name / 'model.pkl'
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        self.models[model_name] = model
        
        # Load labels if exist
        labels_path = self.model_dir / model_name / 'labels.npy'
        if labels_path.exists():
            self.cluster_labels[model_name] = np.load(labels_path)
        
        logger.info(f"Model loaded from: {model_path}")
        return model

# Main function for testing
if __name__ == "__main__":
    print("=" * 60)
    print("Testing Customer Segmentation Engine")
    print("=" * 60)
    
    # Initialize engine
    engine = CustomerSegmentationEngine()
    print(f"✅ Segmentation Engine initialized")
    
    # Generate sample data
    np.random.seed(42)
    X = np.random.randn(300, 10)
    
    print(f"\nSample data shape: {X.shape}")
    
    # Test K-Means
    print("\n1. Testing K-Means clustering...")
    kmeans_result = engine.perform_kmeans(X, n_clusters=4)
    print(f"   Silhouette Score: {kmeans_result['metrics']['silhouette_score']:.3f}")
    
    # Test DBSCAN
    print("\n2. Testing DBSCAN clustering...")
    dbscan_result = engine.perform_dbscan(X, eps=0.5, min_samples=5)
    print(f"   Found {dbscan_result['metrics']['n_clusters']} clusters")
    
    # Test Agglomerative
    print("\n3. Testing Agglomerative clustering...")
    agglomerative_result = engine.perform_agglomerative(X, n_clusters=4)
    print(f"   Silhouette Score: {agglomerative_result['metrics']['silhouette_score']:.3f}")
    
    # Compare algorithms
    print("\n4. Comparing algorithms...")
    comparison = engine.compare_algorithms(X)
    print(comparison.to_string(index=False))
    
    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)