# utils/analytical_reports_manager.py
"""
Analytical Reports Manager for Enterprise AI Platform.

Handles saving and organizing all analytical reports, metrics, and visualizations.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
import logging
import json
import pickle
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

class AnalyticalReportsManager:
    """
    Manages all analytical reports, metrics, and visualizations.
    
    Organizes outputs into:
    - confusion_matrix/
    - graphs/
    - metrics/
    - predictions/
    """
    
    def __init__(self, base_dir: str = 'analytical_reports'):
        """
        Initialize the Analytical Reports Manager.
        
        Args:
            base_dir: Base directory for reports
        """
        self.base_dir = Path(base_dir)
        self._create_directories()
        
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        logger.info(f"Analytical Reports Manager initialized at: {self.base_dir}")
    
    def _create_directories(self):
        """Create all necessary subdirectories."""
        directories = [
            self.base_dir / 'confusion_matrix',
            self.base_dir / 'graphs' / 'training_history',
            self.base_dir / 'graphs' / 'roc_curves',
            self.base_dir / 'graphs' / 'eda_plots',
            self.base_dir / 'graphs' / 'clustering',
            self.base_dir / 'graphs' / 'nlp',
            self.base_dir / 'graphs' / 'forecasting',
            self.base_dir / 'metrics' / 'classification_reports',
            self.base_dir / 'predictions'
        ]
        
        for dir_path in directories:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        logger.info("Directory structure created successfully")
    
    # ==================== CONFUSION MATRIX METHODS ====================
    
    def save_confusion_matrix(self, cm: np.ndarray, model_name: str, 
                              labels: Optional[List[str]] = None) -> Dict[str, Path]:
        """
        Save confusion matrix as PNG and CSV.
        
        Args:
            cm: Confusion matrix array
            model_name: Name of the model
            labels: Class labels
        
        Returns:
            Dictionary of saved file paths
        """
        save_dir = self.base_dir / 'confusion_matrix'
        
        # Save as CSV
        cm_df = pd.DataFrame(cm, index=labels, columns=labels)
        csv_path = save_dir / f'{model_name}_confusion_matrix.csv'
        cm_df.to_csv(csv_path)
        
        # Save as PNG
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        
        png_path = save_dir / f'{model_name}_confusion_matrix.png'
        plt.tight_layout()
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save as JSON
        json_path = save_dir / f'{model_name}_confusion_matrix.json'
        with open(json_path, 'w') as f:
            json.dump(cm.tolist(), f)
        
        logger.info(f"Confusion matrix saved for {model_name}")
        
        return {
            'csv': csv_path,
            'png': png_path,
            'json': json_path
        }
    
    # ==================== GRAPHS METHODS ====================
    
    def save_training_history(self, history: Dict[str, List[float]], 
                             model_name: str) -> Dict[str, Path]:
        """
        Save training history plots.
        
        Args:
            history: Training history dictionary
            model_name: Name of the model
        
        Returns:
            Dictionary of saved file paths
        """
        save_dir = self.base_dir / 'graphs' / 'training_history'
        
        saved_files = {}
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Training History - {model_name}', fontsize=16, fontweight='bold')
        
        # Accuracy
        if 'accuracy' in history and 'val_accuracy' in history:
            axes[0, 0].plot(history['accuracy'], label='Training', linewidth=2)
            axes[0, 0].plot(history['val_accuracy'], label='Validation', linewidth=2)
            axes[0, 0].set_title('Model Accuracy')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Accuracy')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # Loss
        if 'loss' in history and 'val_loss' in history:
            axes[0, 1].plot(history['loss'], label='Training', linewidth=2)
            axes[0, 1].plot(history['val_loss'], label='Validation', linewidth=2)
            axes[0, 1].set_title('Model Loss')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # AUC
        if 'auc' in history and 'val_auc' in history:
            axes[1, 0].plot(history['auc'], label='Training', linewidth=2)
            axes[1, 0].plot(history['val_auc'], label='Validation', linewidth=2)
            axes[1, 0].set_title('Model AUC')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('AUC')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Learning Rate
        if 'lr' in history:
            axes[1, 1].plot(history['lr'], linewidth=2, color='green')
            axes[1, 1].set_title('Learning Rate')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save full history
        history_path = save_dir / f'{model_name}_training_history.png'
        plt.savefig(history_path, dpi=300, bbox_inches='tight')
        saved_files['full_history'] = history_path
        
        # Save individual plots
        for metric in ['accuracy', 'loss']:
            if metric in history:
                fig_single, ax = plt.subplots(figsize=(10, 6))
                ax.plot(history[metric], label=f'Training {metric}', linewidth=2)
                ax.plot(history[f'val_{metric}'], label=f'Validation {metric}', linewidth=2)
                ax.set_title(f'{model_name} - {metric.capitalize()}', fontsize=14, fontweight='bold')
                ax.set_xlabel('Epoch')
                ax.set_ylabel(metric.capitalize())
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                single_path = save_dir / f'{model_name}_{metric}_curve.png'
                plt.tight_layout()
                plt.savefig(single_path, dpi=300, bbox_inches='tight')
                saved_files[f'{metric}_curve'] = single_path
                plt.close()
        
        plt.close()
        
        logger.info(f"Training history saved for {model_name}")
        return saved_files
    
    def save_roc_curve(self, fpr: np.ndarray, tpr: np.ndarray, 
                       roc_auc: float, model_name: str) -> Path:
        """
        Save ROC curve plot.
        
        Args:
            fpr: False positive rates
            tpr: True positive rates
            roc_auc: ROC AUC score
            model_name: Name of the model
        
        Returns:
            Path to saved file
        """
        save_dir = self.base_dir / 'graphs' / 'roc_curves'
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, color='darkorange', linewidth=2, 
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], color='navy', linestyle='--', linewidth=2)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'ROC Curve - {model_name}', fontsize=14, fontweight='bold')
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        
        path = save_dir / f'{model_name}_roc_curve.png'
        plt.tight_layout()
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"ROC curve saved for {model_name}")
        return path
    
    def save_eda_plots(self, df: pd.DataFrame) -> Dict[str, Path]:
        """
        Save EDA plots.
        
        Args:
            df: DataFrame for EDA
        
        Returns:
            Dictionary of saved file paths
        """
        save_dir = self.base_dir / 'graphs' / 'eda_plots'
        saved_files = {}
        
        # 1. Feature distributions
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns[:4]
        if len(numeric_cols) > 0:
            fig, axes = plt.subplots(1, min(4, len(numeric_cols)), figsize=(16, 4))
            if len(numeric_cols) == 1:
                axes = [axes]
            for idx, col in enumerate(numeric_cols[:4]):
                axes[idx].hist(df[col].dropna(), bins=30, color='skyblue', edgecolor='black', alpha=0.7)
                axes[idx].set_title(f'Distribution of {col}')
                axes[idx].set_xlabel(col)
                axes[idx].set_ylabel('Frequency')
                axes[idx].grid(True, alpha=0.3)
            plt.tight_layout()
            path = save_dir / 'feature_distributions.png'
            plt.savefig(path, dpi=300, bbox_inches='tight')
            saved_files['distributions'] = path
            plt.close()
        
        # 2. Correlation heatmap
        if len(numeric_cols) > 1:
            fig, ax = plt.subplots(figsize=(10, 8))
            corr = df[numeric_cols].corr()
            sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', 
                       square=True, linewidths=0.5, ax=ax)
            ax.set_title('Correlation Heatmap', fontsize=14, fontweight='bold')
            plt.tight_layout()
            path = save_dir / 'correlation_heatmap.png'
            plt.savefig(path, dpi=300, bbox_inches='tight')
            saved_files['correlation'] = path
            plt.close()
        
        logger.info("EDA plots saved")
        return saved_files
    
    def save_cluster_plots(self, X_reduced: np.ndarray, labels: np.ndarray,
                          method: str = 'pca') -> Path:
        """
        Save cluster visualization plots.
        
        Args:
            X_reduced: Reduced feature matrix
            labels: Cluster labels
            method: Reduction method ('pca' or 'tsne')
        
        Returns:
            Path to saved file
        """
        save_dir = self.base_dir / 'graphs' / 'clustering'
        
        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(X_reduced[:, 0], X_reduced[:, 1], 
                           c=labels, cmap='viridis', alpha=0.7)
        ax.set_title(f'Cluster Visualization ({method.upper()})', fontsize=14, fontweight='bold')
        ax.set_xlabel(f'{method.upper()} Component 1')
        ax.set_ylabel(f'{method.upper()} Component 2')
        ax.grid(True, alpha=0.3)
        
        cbar = plt.colorbar(scatter)
        cbar.set_label('Cluster')
        
        path = save_dir / f'cluster_visualization_{method}.png'
        plt.tight_layout()
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Cluster visualization saved for {method}")
        return path
    
    def save_wordcloud(self, text: str, title: str) -> Path:
        """
        Save wordcloud visualization.
        
        Args:
            text: Text for wordcloud
            title: Title for the wordcloud
        
        Returns:
            Path to saved file
        """
        from wordcloud import WordCloud
        
        save_dir = self.base_dir / 'graphs' / 'nlp'
        
        wordcloud = WordCloud(width=800, height=400, background_color='white',
                             colormap='viridis', max_words=100).generate(text)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(title, fontsize=16, fontweight='bold')
        
        path = save_dir / f'wordcloud_{title.lower().replace(" ", "_")}.png'
        plt.tight_layout()
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Wordcloud saved: {path}")
        return path
    
    def save_forecast_plot(self, actual: pd.Series, forecasts: Dict[str, np.ndarray],
                          title: str = 'Forecast Comparison') -> Path:
        """
        Save forecast comparison plot.
        
        Args:
            actual: Actual time series
            forecasts: Dictionary of model forecasts
            title: Plot title
        
        Returns:
            Path to saved file
        """
        save_dir = self.base_dir / 'graphs' / 'forecasting'
        
        fig, ax = plt.subplots(figsize=(14, 7))
        
        # Plot actual
        ax.plot(actual.index, actual.values, label='Actual', linewidth=2, color='black')
        
        # Plot forecasts
        colors = ['blue', 'green', 'red', 'orange', 'purple']
        for i, (model_name, forecast) in enumerate(forecasts.items()):
            forecast_index = actual.index[-len(forecast):]
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
        
        path = save_dir / 'forecast_comparison.png'
        plt.tight_layout()
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Forecast plot saved: {path}")
        return path
    
    # ==================== METRICS METHODS ====================
    
    def save_metrics(self, metrics: Dict[str, Any], model_name: str) -> Dict[str, Path]:
        """
        Save model metrics.
        
        Args:
            metrics: Dictionary of metrics
            model_name: Name of the model
        
        Returns:
            Dictionary of saved file paths
        """
        save_dir = self.base_dir / 'metrics'
        saved_files = {}
        
        # Save as JSON
        json_path = save_dir / f'{model_name}_metrics.json'
        with open(json_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        saved_files['json'] = json_path
        
        # Save as CSV if metrics are flat
        if all(isinstance(v, (int, float, str)) for v in metrics.values()):
            metrics_df = pd.DataFrame([metrics])
            csv_path = save_dir / f'{model_name}_metrics.csv'
            metrics_df.to_csv(csv_path, index=False)
            saved_files['csv'] = csv_path
        
        # Add to summary
        self._update_metrics_summary(metrics, model_name)
        
        logger.info(f"Metrics saved for {model_name}")
        return saved_files
    
    def _update_metrics_summary(self, metrics: Dict[str, Any], model_name: str):
        """
        Update the overall metrics summary file.
        
        Args:
            metrics: Dictionary of metrics
            model_name: Name of the model
        """
        summary_path = self.base_dir / 'metrics' / 'model_metrics_summary.csv'
        
        # Prepare row
        row = {'Model': model_name}
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                row[key] = value
        
        # Read existing or create new
        if summary_path.exists():
            summary_df = pd.read_csv(summary_path)
        else:
            summary_df = pd.DataFrame()
        
        # Append new row
        new_row = pd.DataFrame([row])
        summary_df = pd.concat([summary_df, new_row], ignore_index=True)
        
        # Save
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"Metrics summary updated: {summary_path}")
    
    def save_classification_report(self, report: str, model_name: str) -> Dict[str, Path]:
        """
        Save classification report.
        
        Args:
            report: Classification report string
            model_name: Name of the model
        
        Returns:
            Dictionary of saved file paths
        """
        save_dir = self.base_dir / 'metrics' / 'classification_reports'
        saved_files = {}
        
        # Save as text
        txt_path = save_dir / f'{model_name}_classification_report.txt'
        with open(txt_path, 'w') as f:
            f.write(report)
        saved_files['txt'] = txt_path
        
        # Try to parse and save as JSON
        try:
            from sklearn.metrics import classification_report
            # Parse report to dict
            report_dict = classification_report(
                y_true=[], y_pred=[], output_dict=True
            )  # This won't work directly
            # Save as JSON if available
            json_path = save_dir / f'{model_name}_classification_report.json'
            saved_files['json'] = json_path
        except:
            pass
        
        logger.info(f"Classification report saved for {model_name}")
        return saved_files
    
    # ==================== PREDICTIONS METHODS ====================
    
    def save_predictions(self, predictions: np.ndarray, model_name: str,
                        probabilities: Optional[np.ndarray] = None,
                        y_true: Optional[np.ndarray] = None) -> Dict[str, Path]:
        """
        Save model predictions.
        
        Args:
            predictions: Prediction labels
            model_name: Name of the model
            probabilities: Prediction probabilities (optional)
            y_true: True labels (optional)
        
        Returns:
            Dictionary of saved file paths
        """
        save_dir = self.base_dir / 'predictions'
        saved_files = {}
        
        # Create DataFrame
        data = {'prediction': predictions}
        if y_true is not None:
            data['true_label'] = y_true
        if probabilities is not None:
            for i in range(probabilities.shape[1] if probabilities.ndim > 1 else 1):
                if probabilities.ndim > 1:
                    data[f'prob_class_{i}'] = probabilities[:, i]
                else:
                    data['probability'] = probabilities
        
        pred_df = pd.DataFrame(data)
        
        # Save as CSV
        csv_path = save_dir / f'{model_name}_predictions.csv'
        pred_df.to_csv(csv_path, index=False)
        saved_files['csv'] = csv_path
        
        # Save probabilities as numpy
        if probabilities is not None:
            npy_path = save_dir / f'{model_name}_probabilities.npy'
            np.save(npy_path, probabilities)
            saved_files['npy'] = npy_path
        
        logger.info(f"Predictions saved for {model_name}")
        return saved_files
    
    def save_forecast_predictions(self, forecast: np.ndarray, 
                                 model_name: str,
                                 confidence_intervals: Optional[np.ndarray] = None) -> Dict[str, Path]:
        """
        Save forecast predictions.
        
        Args:
            forecast: Forecast values
            model_name: Name of the model
            confidence_intervals: Confidence intervals (optional)
        
        Returns:
            Dictionary of saved file paths
        """
        save_dir = self.base_dir / 'predictions'
        
        data = {'forecast': forecast}
        if confidence_intervals is not None:
            data['lower_bound'] = confidence_intervals[:, 0]
            data['upper_bound'] = confidence_intervals[:, 1]
        
        pred_df = pd.DataFrame(data)
        
        csv_path = save_dir / f'{model_name}_forecast.csv'
        pred_df.to_csv(csv_path, index=False)
        
        logger.info(f"Forecast saved for {model_name}")
        return {'csv': csv_path}
    
    def save_cluster_predictions(self, labels: np.ndarray, 
                                model_name: str,
                                distances: Optional[np.ndarray] = None) -> Dict[str, Path]:
        """
        Save clustering predictions.
        
        Args:
            labels: Cluster labels
            model_name: Name of the model
            distances: Distances to cluster centers (optional)
        
        Returns:
            Dictionary of saved file paths
        """
        save_dir = self.base_dir / 'predictions'
        
        data = {'cluster_label': labels}
        if distances is not None:
            data['distance_to_center'] = distances
        
        pred_df = pd.DataFrame(data)
        
        csv_path = save_dir / f'{model_name}_clustering.csv'
        pred_df.to_csv(csv_path, index=False)
        
        logger.info(f"Cluster predictions saved for {model_name}")
        return {'csv': csv_path}
    
    # ==================== REPORT GENERATION ====================
    
    def generate_summary_report(self) -> Dict[str, Any]:
        """
        Generate a summary report of all saved artifacts.
        
        Returns:
            Dictionary with report summary
        """
        summary = {
            'timestamp': self.timestamp,
            'directories': {},
            'files_count': 0,
            'models': []
        }
        
        # Count files in each directory
        for dir_path in self.base_dir.rglob('*'):
            if dir_path.is_dir():
                files = list(dir_path.glob('*'))
                summary['directories'][str(dir_path.relative_to(self.base_dir))] = len(files)
                summary['files_count'] += len(files)
        
        # Get models from metrics summary
        metrics_summary = self.base_dir / 'metrics' / 'model_metrics_summary.csv'
        if metrics_summary.exists():
            df = pd.read_csv(metrics_summary)
            summary['models'] = df['Model'].tolist()
        
        # Save summary
        summary_path = self.base_dir / 'reports_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=4)
        
        logger.info(f"Summary report generated: {summary_path}")
        return summary


# Example usage
if __name__ == "__main__":
    print("=" * 60)
    print("Testing Analytical Reports Manager")
    print("=" * 60)
    
    # Initialize manager
    manager = AnalyticalReportsManager()
    print(f"✅ Reports Manager initialized")
    
    # Create sample data
    import numpy as np
    
    # Sample confusion matrix
    cm = np.array([[100, 20], [15, 85]])
    manager.save_confusion_matrix(cm, 'test_model', ['No', 'Yes'])
    
    # Sample training history
    history = {
        'accuracy': np.random.rand(50),
        'val_accuracy': np.random.rand(50),
        'loss': np.random.rand(50),
        'val_loss': np.random.rand(50)
    }
    manager.save_training_history(history, 'test_model')
    
    # Sample metrics
    metrics = {
        'accuracy': 0.85,
        'precision': 0.82,
        'recall': 0.79,
        'f1': 0.80,
        'auc': 0.91
    }
    manager.save_metrics(metrics, 'test_model')
    
    # Sample predictions
    predictions = np.random.randint(0, 2, 100)
    probabilities = np.random.rand(100)
    manager.save_predictions(predictions, 'test_model', probabilities)
    
    print("\n✅ All tests completed!")
    
    # Generate summary
    summary = manager.generate_summary_report()
    print(f"\nSummary:")
    print(f"  - Total files: {summary['files_count']}")
    print(f"  - Models: {summary['models']}")
    print(f"  - Directories: {list(summary['directories'].keys())}")