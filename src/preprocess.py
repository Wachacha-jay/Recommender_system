"""
Data preprocessing utilities for recommendation systems.
Handles train/test splits, normalization, and data preparation.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional
from sklearn.model_selection import train_test_split


class DataSplitter:
    """Handle various train/test split strategies."""
    
    @staticmethod
    def random_split(
        ratings: pd.DataFrame,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Random train/test split.
        
        Args:
            ratings: DataFrame with ratings
            test_size: Proportion of data for test set
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (train_df, test_df)
        """
        train, test = train_test_split(
            ratings,
            test_size=test_size,
            random_state=random_state
        )
        
        print(f"Random split: {len(train):,} train, {len(test):,} test")
        return train, test
    
    @staticmethod
    def temporal_split(
        ratings: pd.DataFrame,
        test_size: float = 0.2
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Temporal train/test split based on timestamp.
        
        Args:
            ratings: DataFrame with ratings (must have timestamp column)
            test_size: Proportion of data for test set
            
        Returns:
            Tuple of (train_df, test_df)
        """
        if 'timestamp' not in ratings.columns:
            raise ValueError("Ratings must have 'timestamp' column for temporal split")
        
        # Sort by timestamp
        sorted_ratings = ratings.sort_values('timestamp')
        
        # Split point
        split_idx = int(len(sorted_ratings) * (1 - test_size))
        
        train = sorted_ratings.iloc[:split_idx]
        test = sorted_ratings.iloc[split_idx:]
        
        print(f"Temporal split: {len(train):,} train, {len(test):,} test")
        return train, test
    
    @staticmethod
    def user_based_split(
        ratings: pd.DataFrame,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split ensuring each user has data in both train and test.
        
        Args:
            ratings: DataFrame with ratings
            test_size: Proportion of each user's ratings for test
            random_state: Random seed
            
        Returns:
            Tuple of (train_df, test_df)
        """
        train_list = []
        test_list = []
        
        np.random.seed(random_state)
        
        for user_id, user_ratings in ratings.groupby('user_id'):
            if len(user_ratings) < 2:
                # If user has only 1 rating, put in train
                train_list.append(user_ratings)
                continue
            
            # Shuffle user's ratings
            user_ratings = user_ratings.sample(frac=1, random_state=random_state)
            
            # Split
            n_test = max(1, int(len(user_ratings) * test_size))
            test_list.append(user_ratings.iloc[:n_test])
            train_list.append(user_ratings.iloc[n_test:])
        
        train = pd.concat(train_list, ignore_index=True)
        test = pd.concat(test_list, ignore_index=True)
        
        print(f"User-based split: {len(train):,} train, {len(test):,} test")
        print(f"All {ratings['user_id'].nunique()} users have data in both sets")
        
        return train, test
    
    @staticmethod
    def leave_one_out(
        ratings: pd.DataFrame,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Leave-one-out split: one random rating per user for test.
        
        Args:
            ratings: DataFrame with ratings
            random_state: Random seed
            
        Returns:
            Tuple of (train_df, test_df)
        """
        train_list = []
        test_list = []
        
        np.random.seed(random_state)
        
        for user_id, user_ratings in ratings.groupby('user_id'):
            if len(user_ratings) == 1:
                train_list.append(user_ratings)
                continue
            
            # Sample one rating for test
            test_sample = user_ratings.sample(n=1, random_state=random_state)
            train_sample = user_ratings.drop(test_sample.index)
            
            train_list.append(train_sample)
            test_list.append(test_sample)
        
        train = pd.concat(train_list, ignore_index=True)
        test = pd.concat(test_list, ignore_index=True)
        
        print(f"Leave-one-out split: {len(train):,} train, {len(test):,} test")
        
        return train, test


class DataPreprocessor:
    """Preprocess rating data for recommendation models."""
    
    def __init__(self, min_user_ratings: int = 5, min_item_ratings: int = 5):
        """
        Initialize preprocessor.
        
        Args:
            min_user_ratings: Minimum ratings per user
            min_item_ratings: Minimum ratings per item
        """
        self.min_user_ratings = min_user_ratings
        self.min_item_ratings = min_item_ratings
        
    def filter_sparse_users_items(
        self, 
        ratings: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Filter out users and items with too few ratings.
        
        Args:
            ratings: DataFrame with ratings
            
        Returns:
            Filtered DataFrame
        """
        original_size = len(ratings)
        
        # Iteratively filter until stable
        prev_size = 0
        while len(ratings) != prev_size:
            prev_size = len(ratings)
            
            # Filter users
            user_counts = ratings.groupby('user_id').size()
            valid_users = user_counts[user_counts >= self.min_user_ratings].index
            ratings = ratings[ratings['user_id'].isin(valid_users)]
            
            # Filter items
            item_counts = ratings.groupby('item_id').size()
            valid_items = item_counts[item_counts >= self.min_item_ratings].index
            ratings = ratings[ratings['item_id'].isin(valid_items)]
        
        print(f"Filtered: {original_size:,} -> {len(ratings):,} ratings")
        print(f"Users: {ratings['user_id'].nunique():,}")
        print(f"Items: {ratings['item_id'].nunique():,}")
        
        return ratings
    
    def normalize_ratings(
        self,
        ratings: pd.DataFrame,
        method: str = 'mean_centering'
    ) -> pd.DataFrame:
        """
        Normalize rating values.
        
        Args:
            ratings: DataFrame with ratings
            method: 'mean_centering', 'z_score', or 'min_max'
            
        Returns:
            DataFrame with normalized ratings
        """
        ratings = ratings.copy()
        
        if method == 'mean_centering':
            # Center ratings around user mean
            user_means = ratings.groupby('user_id')['rating'].transform('mean')
            ratings['normalized_rating'] = ratings['rating'] - user_means
            
        elif method == 'z_score':
            # Z-score normalization per user
            user_means = ratings.groupby('user_id')['rating'].transform('mean')
            user_stds = ratings.groupby('user_id')['rating'].transform('std')
            ratings['normalized_rating'] = (ratings['rating'] - user_means) / (user_stds + 1e-8)
            
        elif method == 'min_max':
            # Min-max scaling to [0, 1]
            min_rating = ratings['rating'].min()
            max_rating = ratings['rating'].max()
            ratings['normalized_rating'] = (ratings['rating'] - min_rating) / (max_rating - min_rating)
        
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        print(f"Applied {method} normalization")
        return ratings
    
    def create_negative_samples(
        self,
        ratings: pd.DataFrame,
        n_negative: int = 4,
        random_state: int = 42
    ) -> pd.DataFrame:
        """
        Create negative samples for implicit feedback learning.
        
        Args:
            ratings: DataFrame with positive interactions
            n_negative: Number of negative samples per positive
            random_state: Random seed
            
        Returns:
            DataFrame with positive and negative samples
        """
        np.random.seed(random_state)
        
        # Get all items
        all_items = set(ratings['item_id'].unique())
        
        negative_samples = []
        
        for user_id, user_ratings in ratings.groupby('user_id'):
            # Items user has interacted with
            positive_items = set(user_ratings['item_id'])
            
            # Candidate negative items
            negative_items = list(all_items - positive_items)
            
            if len(negative_items) == 0:
                continue
            
            # Sample negative items
            n_sample = min(n_negative * len(positive_items), len(negative_items))
            sampled_negatives = np.random.choice(negative_items, size=n_sample, replace=False)
            
            for item_id in sampled_negatives:
                negative_samples.append({
                    'user_id': user_id,
                    'item_id': item_id,
                    'rating': 0,
                    'label': 0
                })
        
        # Add label to original ratings
        ratings = ratings.copy()
        ratings['label'] = 1
        
        # Combine
        negative_df = pd.DataFrame(negative_samples)
        combined = pd.concat([ratings, negative_df], ignore_index=True)
        
        print(f"Created {len(negative_samples):,} negative samples")
        print(f"Total samples: {len(combined):,}")
        
        return combined


def prepare_data_for_training(
    ratings: pd.DataFrame,
    test_size: float = 0.2,
    split_method: str = 'user_based',
    min_user_ratings: int = 5,
    min_item_ratings: int = 5,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Complete data preparation pipeline.
    
    Args:
        ratings: Raw ratings DataFrame
        test_size: Test set proportion
        split_method: 'random', 'temporal', 'user_based', or 'leave_one_out'
        min_user_ratings: Minimum ratings per user
        min_item_ratings: Minimum ratings per item
        random_state: Random seed
        
    Returns:
        Tuple of (train_df, test_df, metadata_dict)
    """
    print("="*60)
    print("Data Preparation Pipeline")
    print("="*60)
    
    # 1. Filter sparse users/items
    preprocessor = DataPreprocessor(min_user_ratings, min_item_ratings)
    filtered_ratings = preprocessor.filter_sparse_users_items(ratings)
    
    # 2. Split data
    splitter = DataSplitter()
    
    if split_method == 'random':
        train, test = splitter.random_split(filtered_ratings, test_size, random_state)
    elif split_method == 'temporal':
        train, test = splitter.temporal_split(filtered_ratings, test_size)
    elif split_method == 'user_based':
        train, test = splitter.user_based_split(filtered_ratings, test_size, random_state)
    elif split_method == 'leave_one_out':
        train, test = splitter.leave_one_out(filtered_ratings, random_state)
    else:
        raise ValueError(f"Unknown split method: {split_method}")
    
    # 3. Collect metadata
    metadata = {
        'n_users': ratings['user_id'].nunique(),
        'n_items': ratings['item_id'].nunique(),
        'n_ratings_train': len(train),
        'n_ratings_test': len(test),
        'min_rating': ratings['rating'].min(),
        'max_rating': ratings['rating'].max(),
        'avg_rating': ratings['rating'].mean(),
        'sparsity': 1 - len(ratings) / (ratings['user_id'].nunique() * ratings['item_id'].nunique()),
    }
    
    print("\n" + "="*60)
    print("Data Preparation Complete")
    print("="*60)
    print(f"Train ratings: {len(train):,}")
    print(f"Test ratings: {len(test):,}")
    print(f"Users: {metadata['n_users']:,}")
    print(f"Items: {metadata['n_items']:,}")
    print(f"Sparsity: {metadata['sparsity']:.4f}")
    
    return train, test, metadata


if __name__ == "__main__":
    # Example usage
    from src.data_loader import MovieLensLoader
    
    loader = MovieLensLoader()
    ratings = loader.load_ratings()
    
    # Prepare data with different split methods
    print("\n\n1. USER-BASED SPLIT")
    train, test, metadata = prepare_data_for_training(
        ratings,
        split_method='user_based',
        test_size=0.2
    )
    
    print("\n\n2. TEMPORAL SPLIT")
    train, test, metadata = prepare_data_for_training(
        ratings,
        split_method='temporal',
        test_size=0.2
    )
    
    print("\n\n3. LEAVE-ONE-OUT")
    train, test, metadata = prepare_data_for_training(
        ratings,
        split_method='leave_one_out'
    )