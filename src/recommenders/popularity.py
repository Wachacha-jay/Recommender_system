"""
Popularity-based recommendation algorithms.
Includes global popularity, time-decay, and random baselines.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from datetime import datetime, timedelta
import sys
sys.path.append('..')
from src.recommenders.base import BaseRecommender


class RandomRecommender(BaseRecommender):
    """Random baseline recommender."""
    
    def __init__(self, random_state: int = 42):
        """
        Initialize random recommender.
        
        Args:
            random_state: Random seed for reproducibility
        """
        super().__init__(name="RandomRecommender")
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)
        
    def fit(self, train_data: pd.DataFrame) -> 'RandomRecommender':
        """
        Fit the random recommender (just stores item list).
        
        Args:
            train_data: Training data
            
        Returns:
            self
        """
        self._create_mappings(train_data)
        self.items = list(self.item_id_map.keys())
        self.avg_rating = train_data['rating'].mean()
        self.is_fitted = True
        
        print(f"{self.name} fitted with {len(self.items)} items")
        return self
    
    def predict(self, user_id: int, item_id: int) -> float:
        """
        Predict random rating.
        
        Args:
            user_id: User ID
            item_id: Item ID
            
        Returns:
            Random rating around average
        """
        return self.avg_rating + self.rng.randn() * 0.5


class PopularityRecommender(BaseRecommender):
    """Global popularity-based recommender."""
    
    def __init__(self, method: str = 'count'):
        """
        Initialize popularity recommender.
        
        Args:
            method: 'count' (# of ratings) or 'average' (avg rating)
        """
        super().__init__(name="PopularityRecommender")
        self.method = method
        self.popularity_scores = None
        
    def fit(self, train_data: pd.DataFrame) -> 'PopularityRecommender':
        """
        Fit popularity recommender.
        
        Args:
            train_data: Training data
            
        Returns:
            self
        """
        self._create_mappings(train_data)
        
        if self.method == 'count':
            # Popularity = number of ratings
            self.popularity_scores = train_data.groupby('item_id').size().to_dict()
            
        elif self.method == 'average':
            # Popularity = average rating
            self.popularity_scores = train_data.groupby('item_id')['rating'].mean().to_dict()
            
        elif self.method == 'weighted':
            # Weighted: avg_rating * log(count + 1)
            item_stats = train_data.groupby('item_id').agg({
                'rating': ['mean', 'count']
            })
            item_stats.columns = ['avg_rating', 'count']
            item_stats['weighted_score'] = (
                item_stats['avg_rating'] * np.log1p(item_stats['count'])
            )
            self.popularity_scores = item_stats['weighted_score'].to_dict()
        
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        # Normalize scores
        max_score = max(self.popularity_scores.values())
        self.popularity_scores = {
            k: v / max_score for k, v in self.popularity_scores.items()
        }
        
        self.is_fitted = True
        print(f"{self.name} fitted with method='{self.method}'")
        return self
    
    def predict(self, user_id: int, item_id: int) -> float:
        """
        Predict based on item popularity.
        
        Args:
            user_id: User ID (not used)
            item_id: Item ID
            
        Returns:
            Popularity score
        """
        return self.popularity_scores.get(item_id, 0.0)


class TimeDecayPopularityRecommender(BaseRecommender):
    """Popularity recommender with time decay."""
    
    def __init__(self, decay_rate: float = 0.95, time_unit: str = 'days'):
        """
        Initialize time-decay popularity recommender.
        
        Args:
            decay_rate: Exponential decay rate (< 1)
            time_unit: 'days', 'weeks', or 'months'
        """
        super().__init__(name="TimeDecayPopularityRecommender")
        self.decay_rate = decay_rate
        self.time_unit = time_unit
        self.popularity_scores = None
        self.reference_time = None
        
    def fit(self, train_data: pd.DataFrame) -> 'TimeDecayPopularityRecommender':
        """
        Fit time-decay popularity recommender.
        
        Args:
            train_data: Training data with timestamp column
            
        Returns:
            self
        """
        if 'timestamp' not in train_data.columns:
            raise ValueError("train_data must have 'timestamp' column")
        
        self._create_mappings(train_data)
        
        # Reference time is the latest timestamp
        self.reference_time = train_data['timestamp'].max()
        
        # Calculate time-decayed popularity
        popularity_scores = {}
        
        for item_id, group in train_data.groupby('item_id'):
            # Calculate time differences
            if self.time_unit == 'days':
                time_diffs = (self.reference_time - group['timestamp']) / 86400
            elif self.time_unit == 'weeks':
                time_diffs = (self.reference_time - group['timestamp']) / (86400 * 7)
            elif self.time_unit == 'months':
                time_diffs = (self.reference_time - group['timestamp']) / (86400 * 30)
            else:
                raise ValueError(f"Unknown time_unit: {self.time_unit}")
            
            # Apply exponential decay
            decay_weights = self.decay_rate ** time_diffs
            
            # Popularity = sum of decayed weights
            popularity_scores[item_id] = decay_weights.sum()
        
        # Normalize
        max_score = max(popularity_scores.values())
        self.popularity_scores = {
            k: v / max_score for k, v in popularity_scores.items()
        }
        
        self.is_fitted = True
        print(f"{self.name} fitted with decay_rate={self.decay_rate}")
        return self
    
    def predict(self, user_id: int, item_id: int) -> float:
        """
        Predict based on time-decayed popularity.
        
        Args:
            user_id: User ID (not used)
            item_id: Item ID
            
        Returns:
            Time-decayed popularity score
        """
        return self.popularity_scores.get(item_id, 0.0)


class TrendingRecommender(BaseRecommender):
    """Recommender based on recent trending items."""
    
    def __init__(self, window_days: int = 30):
        """
        Initialize trending recommender.
        
        Args:
            window_days: Number of recent days to consider
        """
        super().__init__(name="TrendingRecommender")
        self.window_days = window_days
        self.trending_scores = None
        
    def fit(self, train_data: pd.DataFrame) -> 'TrendingRecommender':
        """
        Fit trending recommender.
        
        Args:
            train_data: Training data with timestamp
            
        Returns:
            self
        """
        if 'timestamp' not in train_data.columns:
            raise ValueError("train_data must have 'timestamp' column")
        
        self._create_mappings(train_data)
        
        # Get recent data
        max_timestamp = train_data['timestamp'].max()
        window_start = max_timestamp - (self.window_days * 86400)
        
        recent_data = train_data[train_data['timestamp'] >= window_start]
        
        # Calculate trending score (velocity)
        # Compare recent popularity to historical
        recent_counts = recent_data.groupby('item_id').size()
        total_counts = train_data.groupby('item_id').size()
        
        trending_scores = {}
        for item_id in total_counts.index:
            recent = recent_counts.get(item_id, 0)
            total = total_counts[item_id]
            
            # Trending score = recent_ratio * sqrt(recent_count)
            recent_ratio = recent / (total + 1)
            trending_scores[item_id] = recent_ratio * np.sqrt(recent + 1)
        
        # Normalize
        max_score = max(trending_scores.values()) if trending_scores else 1.0
        self.trending_scores = {
            k: v / max_score for k, v in trending_scores.items()
        }
        
        self.is_fitted = True
        print(f"{self.name} fitted with window_days={self.window_days}")
        print(f"Found {len(self.trending_scores)} trending items")
        return self
    
    def predict(self, user_id: int, item_id: int) -> float:
        """
        Predict based on trending score.
        
        Args:
            user_id: User ID (not used)
            item_id: Item ID
            
        Returns:
            Trending score
        """
        return self.trending_scores.get(item_id, 0.0)


if __name__ == "__main__":
    # Example usage
    from src.data_loader import MovieLensLoader
    from src.preprocess import prepare_data_for_training
    
    # Load data
    loader = MovieLensLoader()
    ratings = loader.load_ratings()
    
    # Prepare data
    train, test, _ = prepare_data_for_training(ratings, test_size=0.2)
    
    # Test different popularity recommenders
    print("\n" + "="*60)
    print("Testing Popularity Recommenders")
    print("="*60)
    
    # Random
    print("\n1. Random Recommender")
    random_rec = RandomRecommender()
    random_rec.fit(train)
    recommendations = random_rec.recommend(user_id=1, n=5)
    print(f"Top 5 for user 1: {recommendations}")
    
    # Global popularity
    print("\n2. Popularity Recommender (count)")
    pop_rec = PopularityRecommender(method='count')
    pop_rec.fit(train)
    recommendations = pop_rec.recommend(user_id=1, n=5)
    print(f"Top 5 for user 1: {recommendations}")
    
    # Time-decay
    print("\n3. Time-Decay Popularity")
    time_rec = TimeDecayPopularityRecommender(decay_rate=0.95)
    time_rec.fit(train)
    recommendations = time_rec.recommend(user_id=1, n=5)
    print(f"Top 5 for user 1: {recommendations}")
    
    # Trending
    print("\n4. Trending Recommender")
    trend_rec = TrendingRecommender(window_days=30)
    trend_rec.fit(train)
    recommendations = trend_rec.recommend(user_id=1, n=5)
    print(f"Top 5 for user 1: {recommendations}")