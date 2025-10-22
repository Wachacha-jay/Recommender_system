"""
Recommendation System Package
A modular framework for building and comparing recommendation algorithms.
"""

__version__ = "0.1.0"
__author__ = "Your Name"

from .data_loader import MovieLensLoader, download_movielens
from .preprocess import DataSplitter, DataPreprocessor, prepare_data_for_training
from .evaluation import RecommenderEvaluator

__all__ = [
    'MovieLensLoader',
    'download_movielens',
    'DataSplitter',
    'DataPreprocessor',
    'prepare_data_for_training',
    'RecommenderEvaluator',
]

