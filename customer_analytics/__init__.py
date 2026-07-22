# customer_analytics/__init__.py
"""
Customer Analytics Module for Enterprise AI Platform.

This module provides various analytics capabilities for customer data including:
- Neural Networks for classification
- Sequence Models for time series and anomaly detection
- Segmentation Engines for customer clustering
"""

from .neural_nets import NeuralNetworkEngine
from .sequence_models import SequenceModelEngine
from .segmentation_engines import CustomerSegmentationEngine

__all__ = [
    'NeuralNetworkEngine',
    'SequenceModelEngine', 
    'CustomerSegmentationEngine'
]

__version__ = '1.0.0'