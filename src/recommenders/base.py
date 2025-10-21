"""
Abstract base class for all recommendation algorithms.
Defines the common interface that all recommenders must implement.
"""

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict


class BaseRecommender(ABC):
    """Abstract base class for recommender systems."""
    
    def __init__(self, name: str = "BaseRecommender"):
        """
        Initialize recommender.
        
        Args:
            name: Name of the recommender algorithm
        """
        self.name = name
        self.is_fitted = False
        self.user_id_map = None
        self.item_id_map = None
        self.reverse_user_map = None
        self.reverse_item_map = None
        
    @abstractmethod
    def fit(self, train_data: pd.DataFrame) -> 'BaseRecommender':
        """
        Train the recommender model.
        
        Args:
            train_data: Training data with columns [user_id, item_id, rating]
            
        Returns:
            self
        """
        pass
    
    @abstractmethod
    def predict(self, user_id: int, item_id: int) -> float:
        """
        Predict rating for a user-item pair.
        
        Args:
            user_id: User ID
            item_id: Item ID
            
        Returns:
            Predicted rating score
        """
        pass
    
    def recommend(
        self, 
        user_id: int, 
        n: int = 10,
        exclude_seen: bool = True,
        seen_items: Optional[set] = None
    ) -> List[Tuple[int, float]]:
        """
        Generate top-N recommendations for a user.
        
        Args:
            user_id: User ID
            n: Number of recommendations
            exclude_seen: Whether to exclude items user has seen
            seen_items: Set of item IDs user has already seen
            
        Returns:
            List of (item_id, score) tuples sorted by score descending
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before generating recommendations")
        
        # Get all items
        all_items = list(self.item_id_map.keys())
        
        # Exclude seen items if requested
        if exclude_seen and seen_items is not None:
            all_items = [item for item in all_items if item not in seen_items]
        
        # Predict scores for all items
        scores = []
        for item_id in all_items:
            try:
                score = self.predict(user_id, item_id)
                scores.append((item_id, score))
            except:
                # Skip items that can't be scored
                continue
        
        # Sort by score and return top N
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:n]
    
    def recommend_batch(
        self,
        user_ids: List[int],
        n: int = 10,
        exclude_seen: bool = True
    ) -> Dict[int, List[Tuple[int, float]]]:
        """
        Generate recommendations for multiple users.
        
        Args:
            user_ids: List of user IDs
            n: Number of recommendations per user
            exclude_seen: Whether to exclude seen items
            
        Returns:
            Dictionary mapping user_id to list of recommendations
        """
        recommendations = {}
        for user_id in user_ids:
            recommendations[user_id] = self.recommend(
                user_id, n=n, exclude_seen=exclude_seen
            )
        return recommendations
    
    def _create_mappings(self, train_data: pd.DataFrame) -> None:
        """
        Create ID mappings from original IDs to matrix indices.
        
        Args:
            train_data: Training data
        """
        unique_users = sorted(train_data['user_id'].unique())
        unique_items = sorted(train_data['item_id'].unique())
        
        self.user_id_map = {uid: idx for idx, uid in enumerate(unique_users)}
        self.item_id_map = {iid: idx for idx, iid in enumerate(unique_items)}
        
        self.reverse_user_map = {idx: uid for uid, idx in self.user_id_map.items()}
        self.reverse_item_map = {idx: iid for iid, idx in self.item_id_map.items()}
    
    def _get_user_items(self, train_data: pd.DataFrame) -> Dict[int, set]:
        """
        Get items each user has interacted with.
        
        Args:
            train_data: Training data
            
        Returns:
            Dictionary mapping user_id to set of item_ids
        """
        user_items = {}
        for user_id, group in train_data.groupby('user_id'):
            user_items[user_id] = set(group['item_id'].values)
        return user_items
    
    def get_model_size(self) -> Dict[str, int]:
        """
        Get model size information.
        
        Returns:
            Dictionary with model size metrics
        """
        return {
            'n_users': len(self.user_id_map) if self.user_id_map else 0,
            'n_items': len(self.item_id_map) if self.item_id_map else 0,
        }
    
    def __repr__(self) -> str:
        """String representation of the recommender."""
        fitted_str = "fitted" if self.is_fitted else "not fitted"
        return f"{self.name} ({fitted_str})"
    
    def save(self, filepath: str) -> None:
        """
        Save model to file.
        
        Args:
            filepath: Path to save model
        """
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"Model saved to {filepath}")
    
    @staticmethod
    def load(filepath: str) -> 'BaseRecommender':
        """
        Load model from file.
        
        Args:
            filepath: Path to model file
            
        Returns:
            Loaded recommender model
        """
        import pickle
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded from {filepath}")
        return model