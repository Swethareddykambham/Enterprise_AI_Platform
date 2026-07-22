# core_pipeline/__init__.py
"""
Core Pipeline Module for Enterprise AI Platform.

This module provides unified data transformation and feature engineering pipelines
for the entire AI platform.
"""

from .transformation_pipes import DataTransformationPipeline, FeatureEngineering

__all__ = [
    'DataTransformationPipeline',
    'FeatureEngineering'
]

__version__ = '1.0.0'