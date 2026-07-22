# natural_language/__init__.py
"""
Natural Language Processing Module for Enterprise AI Platform.

This module provides NLP capabilities including:
- Text preprocessing (tokenization, lemmatization, POS tagging)
- Sentiment analysis
- Topic modeling
- Named Entity Recognition (NER)
- Word embeddings (TF-IDF, Word2Vec, GloVe)
"""

from .nlp_pipelines import NLPPipeline, TextPreprocessor, SentimentAnalyzer, TopicModeler

__all__ = [
    'NLPPipeline',
    'TextPreprocessor', 
    'SentimentAnalyzer',
    'TopicModeler'
]

__version__ = '1.0.0'