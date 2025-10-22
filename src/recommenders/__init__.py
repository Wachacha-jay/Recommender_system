"""
Recommender algorithms module.
Contains various recommendation algorithms.
"""

from .base import BaseRecommender
from .popularity import (
    RandomRecommender,
    PopularityRecommender,
    TimeDecayPopularityRecommender,
    TrendingRecommender
)
from .collaborative import (
    UserBasedCF,
    ItemBasedCF,
    MatrixFactorizationSVD,
    AlternatingLeastSquares
)
from .content_based import (
    ContentBasedRecommender,
    HybridRecommender
)

__all__ = [
    # Base
    'BaseRecommender',
    
    # Popularity-based
    'RandomRecommender',
    'PopularityRecommender',
    'TimeDecayPopularityRecommender',
    'TrendingRecommender',
    
    # Collaborative Filtering
    'UserBasedCF',
    'ItemBasedCF',
    'MatrixFactorizationSVD',
    'AlternatingLeastSquares',
    
    # Content-based
    'ContentBasedRecommender',
    'HybridRecommender',
]