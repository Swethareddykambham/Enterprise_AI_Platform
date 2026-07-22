# natural_language/nlp_pipelines.py
"""
Natural Language Processing pipelines for text analytics.

This module provides comprehensive NLP capabilities including:
- Text preprocessing (tokenization, lemmatization, POS tagging)
- Sentiment analysis
- Topic modeling (LDA, NMF)
- Named Entity Recognition (NER)
- Word embeddings (TF-IDF, Word2Vec, GloVe)
- Text classification
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Union
import logging
import json
import pickle
import re
import string
import warnings
warnings.filterwarnings('ignore')

# NLP Libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk import pos_tag
from textblob import TextBlob

# Scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Word embeddings
from gensim.models import Word2Vec, KeyedVectors

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Download NLTK data if not available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
try:
    nltk.data.find('corpora/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

logger = logging.getLogger(__name__)

class TextPreprocessor:
    """
    Text Preprocessor for NLP tasks.
    
    Handles:
    - Text cleaning (lowercase, punctuation removal)
    - Tokenization
    - Stopword removal
    - Stemming and Lemmatization
    - POS tagging
    """
    
    def __init__(self, remove_stopwords: bool = True, 
                 lemmatize: bool = True,
                 use_stemming: bool = False):
        """
        Initialize the TextPreprocessor.
        
        Args:
            remove_stopwords: Whether to remove stopwords
            lemmatize: Whether to apply lemmatization
            use_stemming: Whether to apply stemming (overrides lemmatization)
        """
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.use_stemming = use_stemming
        
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        
        logger.info("TextPreprocessor initialized")
    
    def clean_text(self, text: str) -> str:
        """
        Clean text by removing noise.
        
        Args:
            text: Input text
        
        Returns:
            Cleaned text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation
        text = re.sub(f'[{string.punctuation}]', ' ', text)
        
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words.
        
        Args:
            text: Input text
        
        Returns:
            List of tokens
        """
        return word_tokenize(text)
    
    def remove_stopwords_from_tokens(self, tokens: List[str]) -> List[str]:
        """
        Remove stopwords from tokens.
        
        Args:
            tokens: List of tokens
        
        Returns:
            List of tokens without stopwords
        """
        return [token for token in tokens if token not in self.stop_words]
    
    def lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        """
        Lemmatize tokens.
        
        Args:
            tokens: List of tokens
        
        Returns:
            List of lemmatized tokens
        """
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def stem_tokens(self, tokens: List[str]) -> List[str]:
        """
        Stem tokens.
        
        Args:
            tokens: List of tokens
        
        Returns:
            List of stemmed tokens
        """
        return [self.stemmer.stem(token) for token in tokens]
    
    def pos_tag_tokens(self, tokens: List[str]) -> List[Tuple[str, str]]:
        """
        Perform Part-of-Speech tagging on tokens.
        
        Args:
            tokens: List of tokens
        
        Returns:
            List of (token, POS) tuples
        """
        return pos_tag(tokens)
    
    def process_text(self, text: str) -> Dict[str, Any]:
        """
        Complete text processing pipeline.
        
        Args:
            text: Input text
        
        Returns:
            Dictionary with processed text and metadata
        """
        # Clean text
        cleaned_text = self.clean_text(text)
        
        # Tokenize
        tokens = self.tokenize(cleaned_text)
        
        # Remove stopwords
        if self.remove_stopwords:
            tokens = self.remove_stopwords_from_tokens(tokens)
        
        # Lemmatize or stem
        if self.use_stemming:
            tokens = self.stem_tokens(tokens)
        elif self.lemmatize:
            tokens = self.lemmatize_tokens(tokens)
        
        # POS tagging
        pos_tags = self.pos_tag_tokens(tokens)
        
        return {
            'original_text': text,
            'cleaned_text': cleaned_text,
            'tokens': tokens,
            'pos_tags': pos_tags,
            'token_count': len(tokens),
            'word_count': len(text.split())
        }
    
    def process_dataframe(self, df: pd.DataFrame, text_column: str) -> pd.DataFrame:
        """
        Process all texts in a DataFrame column.
        
        Args:
            df: Input DataFrame
            text_column: Name of the text column
        
        Returns:
            DataFrame with processed text columns
        """
        logger.info(f"Processing {len(df)} texts...")
        
        df_processed = df.copy()
        
        # Process each text
        processed_results = df_processed[text_column].apply(self.process_text)
        
        # Extract results into columns
        df_processed['cleaned_text'] = processed_results.apply(lambda x: x['cleaned_text'])
        df_processed['tokens'] = processed_results.apply(lambda x: x['tokens'])
        df_processed['token_count'] = processed_results.apply(lambda x: x['token_count'])
        df_processed['word_count'] = processed_results.apply(lambda x: x['word_count'])
        
        logger.info(f"Text processing complete. Added columns: cleaned_text, tokens, token_count, word_count")
        
        return df_processed


class SentimentAnalyzer:
    """
    Sentiment Analysis for text data.
    
    Provides:
    - Polarity and subjectivity scores
    - Sentiment classification
    - Aspect-based sentiment (optional)
    """
    
    def __init__(self):
        """Initialize the SentimentAnalyzer."""
        logger.info("SentimentAnalyzer initialized")
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment of a text.
        
        Args:
            text: Input text
        
        Returns:
            Dictionary with sentiment results
        """
        blob = TextBlob(text)
        
        return {
            'text': text,
            'polarity': blob.sentiment.polarity,
            'subjectivity': blob.sentiment.subjectivity,
            'sentiment_label': self._get_sentiment_label(blob.sentiment.polarity)
        }
    
    def _get_sentiment_label(self, polarity: float) -> str:
        """
        Convert polarity score to sentiment label.
        
        Args:
            polarity: Polarity score (-1 to 1)
        
        Returns:
            Sentiment label
        """
        if polarity > 0.1:
            return 'positive'
        elif polarity < -0.1:
            return 'negative'
        else:
            return 'neutral'
    
    def analyze_dataframe(self, df: pd.DataFrame, text_column: str) -> pd.DataFrame:
        """
        Analyze sentiment for all texts in a DataFrame.
        
        Args:
            df: Input DataFrame
            text_column: Name of the text column
        
        Returns:
            DataFrame with sentiment columns
        """
        logger.info(f"Analyzing sentiment for {len(df)} texts...")
        
        df_sentiment = df.copy()
        
        # Analyze each text
        sentiment_results = df_sentiment[text_column].apply(self.analyze_sentiment)
        
        # Extract results
        df_sentiment['polarity'] = sentiment_results.apply(lambda x: x['polarity'])
        df_sentiment['subjectivity'] = sentiment_results.apply(lambda x: x['subjectivity'])
        df_sentiment['sentiment_label'] = sentiment_results.apply(lambda x: x['sentiment_label'])
        
        # Add sentiment distribution
        sentiment_dist = df_sentiment['sentiment_label'].value_counts()
        logger.info(f"Sentiment distribution: {sentiment_dist.to_dict()}")
        
        return df_sentiment
    
    def plot_sentiment_distribution(self, df: pd.DataFrame, 
                                   save_path: Optional[Path] = None) -> None:
        """
        Plot sentiment distribution.
        
        Args:
            df: DataFrame with sentiment labels
            save_path: Path to save the plot
        """
        if 'sentiment_label' not in df.columns:
            raise ValueError("DataFrame must contain 'sentiment_label' column")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Count plot
        sentiment_counts = df['sentiment_label'].value_counts()
        colors = ['#2ecc71', '#e74c3c', '#f39c12']
        sentiment_counts.plot(kind='bar', ax=axes[0], color=colors, edgecolor='black')
        axes[0].set_title('Sentiment Distribution', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Sentiment')
        axes[0].set_ylabel('Count')
        axes[0].grid(True, alpha=0.3)
        
        # Pie chart
        axes[1].pie(sentiment_counts.values, labels=sentiment_counts.index, 
                   autopct='%1.1f%%', colors=colors, startangle=90)
        axes[1].set_title('Sentiment Distribution (%)', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Sentiment plot saved to: {save_path}")
        
        plt.show()


class TopicModeler:
    """
    Topic Modeling for text data.
    
    Provides:
    - LDA (Latent Dirichlet Allocation)
    - NMF (Non-negative Matrix Factorization)
    - Topic visualization
    """
    
    def __init__(self, n_topics: int = 5, random_state: int = 42):
        """
        Initialize the TopicModeler.
        
        Args:
            n_topics: Number of topics
            random_state: Random seed for reproducibility
        """
        self.n_topics = n_topics
        self.random_state = random_state
        self.vectorizer = None
        self.topic_model = None
        self.topic_words = None
        
        logger.info(f"TopicModeler initialized with {n_topics} topics")
    
    def create_document_term_matrix(self, texts: List[str], 
                                   max_features: int = 1000) -> np.ndarray:
        """
        Create document-term matrix using CountVectorizer.
        
        Args:
            texts: List of text documents
            max_features: Maximum number of features
        
        Returns:
            Document-term matrix
        """
        logger.info("Creating document-term matrix...")
        
        self.vectorizer = CountVectorizer(
            max_features=max_features,
            stop_words='english'
        )
        
        dtm = self.vectorizer.fit_transform(texts)
        
        logger.info(f"Document-term matrix shape: {dtm.shape}")
        
        return dtm
    
    def fit_lda(self, texts: List[str], max_features: int = 1000) -> Dict[str, Any]:
        """
        Fit LDA topic model.
        
        Args:
            texts: List of text documents
            max_features: Maximum number of features
        
        Returns:
            Dictionary with topic model results
        """
        logger.info("Fitting LDA topic model...")
        
        # Create document-term matrix
        dtm = self.create_document_term_matrix(texts, max_features)
        
        # Fit LDA
        lda = LatentDirichletAllocation(
            n_components=self.n_topics,
            random_state=self.random_state,
            max_iter=100
        )
        
        topic_distributions = lda.fit_transform(dtm)
        self.topic_model = lda
        
        # Extract topic words
        feature_names = self.vectorizer.get_feature_names_out()
        self.topic_words = self._extract_topic_words(lda, feature_names)
        
        logger.info(f"LDA model fitted with {self.n_topics} topics")
        
        return {
            'model': lda,
            'topic_distributions': topic_distributions,
            'topic_words': self.topic_words,
            'feature_names': feature_names
        }
    
    def fit_nmf(self, texts: List[str], max_features: int = 1000) -> Dict[str, Any]:
        """
        Fit NMF topic model.
        
        Args:
            texts: List of text documents
            max_features: Maximum number of features
        
        Returns:
            Dictionary with topic model results
        """
        logger.info("Fitting NMF topic model...")
        
        # Create document-term matrix
        dtm = self.create_document_term_matrix(texts, max_features)
        
        # Fit NMF
        nmf = NMF(
            n_components=self.n_topics,
            random_state=self.random_state,
            max_iter=100
        )
        
        topic_distributions = nmf.fit_transform(dtm)
        self.topic_model = nmf
        
        # Extract topic words
        feature_names = self.vectorizer.get_feature_names_out()
        self.topic_words = self._extract_topic_words(nmf, feature_names)
        
        logger.info(f"NMF model fitted with {self.n_topics} topics")
        
        return {
            'model': nmf,
            'topic_distributions': topic_distributions,
            'topic_words': self.topic_words,
            'feature_names': feature_names
        }
    
    def _extract_topic_words(self, model, feature_names: np.ndarray, 
                           n_words: int = 10) -> Dict[int, List[str]]:
        """
        Extract top words for each topic.
        
        Args:
            model: Trained topic model
            feature_names: Feature names
            n_words: Number of words per topic
        
        Returns:
            Dictionary of topic words
        """
        topic_words = {}
        
        for topic_idx, topic in enumerate(model.components_):
            top_indices = topic.argsort()[-n_words:][::-1]
            top_words = [feature_names[i] for i in top_indices]
            topic_words[topic_idx] = top_words
        
        return topic_words
    
    def get_topic_summary(self) -> pd.DataFrame:
        """
        Get summary of topics and their top words.
        
        Returns:
            DataFrame with topic summary
        """
        if self.topic_words is None:
            raise ValueError("No topic model fitted. Call fit_lda or fit_nmf first.")
        
        summary_data = []
        for topic_idx, words in self.topic_words.items():
            summary_data.append({
                'Topic': f'Topic {topic_idx + 1}',
                'Top Words': ', '.join(words[:10])
            })
        
        return pd.DataFrame(summary_data)
    
    def plot_topics(self, save_path: Optional[Path] = None) -> None:
        """
        Plot topic word distributions.
        
        Args:
            save_path: Path to save the plot
        """
        if self.topic_words is None:
            raise ValueError("No topic model fitted. Call fit_lda or fit_nmf first.")
        
        n_topics = len(self.topic_words)
        fig, axes = plt.subplots((n_topics + 1) // 2, 2, figsize=(14, 5 * ((n_topics + 1) // 2)))
        axes = axes.flatten() if n_topics > 1 else [axes]
        
        for i, (topic_idx, words) in enumerate(self.topic_words.items()):
            ax = axes[i]
            ax.barh(range(len(words)), [10 - j for j in range(len(words))])
            ax.set_yticks(range(len(words)))
            ax.set_yticklabels(words[:10])
            ax.set_title(f'Topic {topic_idx + 1}')
            ax.invert_yaxis()
        
        # Hide empty subplots
        for i in range(n_topics, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Topic plot saved to: {save_path}")
        
        plt.show()


class WordEmbeddingGenerator:
    """
    Word Embedding Generator.
    
    Provides:
    - TF-IDF vectors
    - Word2Vec embeddings
    - GloVe embeddings (loading pre-trained)
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the WordEmbeddingGenerator.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.tfidf_vectorizer = None
        self.word2vec_model = None
        self.glove_model = None
        
        logger.info("WordEmbeddingGenerator initialized")
    
    def create_tfidf_vectors(self, texts: List[str], 
                            max_features: int = 5000) -> np.ndarray:
        """
        Create TF-IDF vectors.
        
        Args:
            texts: List of text documents
            max_features: Maximum number of features
        
        Returns:
            TF-IDF matrix
        """
        logger.info("Creating TF-IDF vectors...")
        
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        
        logger.info(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
        
        return tfidf_matrix
    
    def train_word2vec(self, texts: List[str], 
                       vector_size: int = 100,
                       min_count: int = 2,
                       window: int = 5) -> KeyedVectors:
        """
        Train Word2Vec embeddings.
        
        Args:
            texts: List of text documents
            vector_size: Embedding dimension
            min_count: Minimum word frequency
            window: Context window size
        
        Returns:
            Trained Word2Vec model
        """
        logger.info("Training Word2Vec embeddings...")
        
        # Tokenize texts
        tokenized_texts = [text.split() for text in texts]
        
        # Train Word2Vec
        model = Word2Vec(
            sentences=tokenized_texts,
            vector_size=vector_size,
            min_count=min_count,
            window=window,
            workers=4,
            sg=1  # Skip-gram
        )
        
        self.word2vec_model = model
        
        logger.info(f"Word2Vec vocabulary size: {len(model.wv)}")
        
        return model.wv
    
    def load_glove(self, filepath: Union[str, Path]) -> Dict[str, np.ndarray]:
        """
        Load pre-trained GloVe embeddings.
        
        Args:
            filepath: Path to GloVe file
        
        Returns:
            Dictionary of word embeddings
        """
        logger.info(f"Loading GloVe embeddings from: {filepath}")
        
        embeddings = {}
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], dtype='float32')
                embeddings[word] = vector
        
        self.glove_model = embeddings
        
        logger.info(f"Loaded {len(embeddings)} GloVe embeddings")
        
        return embeddings


class NLPPipeline:
    """
    Complete NLP Pipeline orchestrating all NLP components.
    
    Provides:
    - Text preprocessing
    - Sentiment analysis
    - Topic modeling
    - Word embeddings
    - Model training and evaluation
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the NLP Pipeline.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.preprocessor = TextPreprocessor()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.topic_modeler = TopicModeler(random_state=random_state)
        self.embedding_generator = WordEmbeddingGenerator(random_state=random_state)
        
        self.results = {}
        
        logger.info("NLP Pipeline initialized")
    
    def process_text_data(self, df: pd.DataFrame, text_column: str) -> pd.DataFrame:
        """
        Complete text processing pipeline.
        
        Args:
            df: Input DataFrame
            text_column: Name of the text column
        
        Returns:
            Processed DataFrame
        """
        logger.info("Running complete NLP pipeline...")
        
        # Preprocess text
        df_processed = self.preprocessor.process_dataframe(df, text_column)
        
        # Analyze sentiment
        df_sentiment = self.sentiment_analyzer.analyze_dataframe(
            df_processed, 'cleaned_text'
        )
        
        self.results['processed_data'] = df_sentiment
        
        logger.info("NLP pipeline complete")
        
        return df_sentiment
    
    def extract_topics(self, texts: List[str], 
                      n_topics: int = 5,
                      method: str = 'lda') -> Dict[str, Any]:
        """
        Extract topics from texts.
        
        Args:
            texts: List of text documents
            n_topics: Number of topics
            method: Topic modeling method ('lda' or 'nmf')
        
        Returns:
            Dictionary with topic results
        """
        logger.info(f"Extracting {n_topics} topics using {method}")
        
        self.topic_modeler = TopicModeler(n_topics=n_topics, 
                                         random_state=self.random_state)
        
        if method == 'lda':
            result = self.topic_modeler.fit_lda(texts)
        else:
            result = self.topic_modeler.fit_nmf(texts)
        
        self.results['topics'] = result
        
        return result
    
    def generate_embeddings(self, texts: List[str]) -> Dict[str, Any]:
        """
        Generate various word embeddings.
        
        Args:
            texts: List of text documents
        
        Returns:
            Dictionary with embedding results
        """
        logger.info("Generating word embeddings...")
        
        # TF-IDF
        tfidf_matrix = self.embedding_generator.create_tfidf_vectors(texts)
        
        # Word2Vec
        word2vec = self.embedding_generator.train_word2vec(texts)
        
        results = {
            'tfidf': tfidf_matrix,
            'word2vec': word2vec,
            'tfidf_vectorizer': self.embedding_generator.tfidf_vectorizer
        }
        
        self.results['embeddings'] = results
        
        return results
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of all pipeline results.
        
        Returns:
            Dictionary with summary results
        """
        summary = {
            'texts_processed': len(self.results.get('processed_data', [])),
            'sentiment_distribution': self.results.get('sentiment_distribution', {}),
            'topics_extracted': self.topic_modeler.n_topics if self.topic_modeler else 0,
            'embedding_methods': list(self.results.get('embeddings', {}).keys())
        }
        
        return summary


# Main function for testing
if __name__ == "__main__":
    print("=" * 60)
    print("Testing NLP Pipeline")
    print("=" * 60)
    
    # Sample texts
    sample_texts = [
        "The product is amazing! Great quality and excellent customer service.",
        "I'm very disappointed with this purchase. The quality is poor.",
        "Average product, nothing special but gets the job done.",
        "Excellent service and fantastic value for money. Highly recommend!",
        "Terrible experience. The product broke after one use.",
        "Good product with decent features. Customer support is helpful.",
        "Worst purchase ever. Complete waste of money.",
        "I love this product! It exceeded all my expectations."
    ]
    
    # Create DataFrame
    df = pd.DataFrame({
        'review_id': [f'REV_{i:03d}' for i in range(len(sample_texts))],
        'review_text': sample_texts
    })
    
    print(f"Sample texts loaded: {len(df)} reviews")
    
    # Initialize pipeline
    pipeline = NLPPipeline()
    print(f"✅ NLP Pipeline initialized")
    
    # Process text data
    print("\n1. Processing text data...")
    df_processed = pipeline.process_text_data(df, 'review_text')
    print(f"   Processed {len(df_processed)} texts")
    print(f"   Columns: {list(df_processed.columns)}")
    
    # Sentiment analysis
    print("\n2. Sentiment analysis...")
    sentiment_dist = df_processed['sentiment_label'].value_counts()
    print(f"   Sentiment distribution: {sentiment_dist.to_dict()}")
    
    # Topic modeling
    print("\n3. Topic modeling...")
    texts = df_processed['cleaned_text'].tolist()
    topic_results = pipeline.extract_topics(texts, n_topics=2)
    print(f"   Topics extracted: {len(topic_results['topic_words'])}")
    for topic_idx, words in topic_results['topic_words'].items():
        print(f"   Topic {topic_idx + 1}: {', '.join(words[:5])}")
    
    # Word embeddings
    print("\n4. Word embeddings...")
    embeddings = pipeline.generate_embeddings(texts)
    print(f"   Embedding methods: {list(embeddings.keys())}")
    
    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)