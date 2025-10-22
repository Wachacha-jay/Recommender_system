"""
Content-based recommendation algorithms.
Uses item features (genres, metadata) to make recommendations.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import sys
sys.path.append('..')
from src.recommenders.base import BaseRecommender


class ContentBasedRecommender(BaseRecommender):
    """Content-based recommender using item features."""
    
    def __init__(self, similarity: str = 'cosine'):
        """
        Initialize content-based recommender.
        
        Args:
            similarity: Similarity metric
        """
        super().__init__(name="ContentBasedRecommender")
        self.similarity_metric = similarity
        self.item_features = None
        self.item_similarity = None
        self.user_profiles = None
        self.movies_data = None
        
    def fit(
        self, 
        train_data: pd.DataFrame,
        movies_data: pd.DataFrame
    ) -> 'ContentBasedRecommender':
        """
        Fit content-based recommender.
        
        Args:
            train_data: Training data with ratings
            movies_data: Movie metadata with genres
            
        Returns:
            self
        """
        self._create_mappings(train_data)
        self.movies_data = movies_data
        
        # Create genre-based features for each item
        genre_features = self._create_genre_features(movies_data)
        
        # Store item features
        self.item_features = genre_features
        
        # Calculate item similarity matrix
        self.item_similarity = cosine_similarity(genre_features)
        
        # Build user profiles based on their rating history
        self.user_profiles = self._build_user_profiles(train_data, genre_features)
        
        self.is_fitted = True
        print(f"{self.name} fitted with {genre_features.shape[1]} features")
        return self
    
    def _create_genre_features(self, movies_data: pd.DataFrame) -> np.ndarray:
        """
        Create feature matrix from movie genres.
        
        Args:
            movies_data: Movie metadata
            
        Returns:
            Feature matrix (n_items × n_features)
        """
        # Convert genres list to string
        movies_data = movies_data.copy()
        movies_data['genres_str'] = movies_data['genres'].apply(
            lambda x: ' '.join(x) if isinstance(x, list) else ''
        )
        
        # Use TF-IDF to create features
        vectorizer = TfidfVectorizer()
        genre_features = vectorizer.fit_transform(movies_data['genres_str'])
        
        # Create mapping from item_id to row index
        item_id_to_idx = {
            row['item_id']: idx 
            for idx, row in movies_data.iterrows()
        }
        
        # Reorder features according to item_id_map
        n_items = len(self.item_id_map)
        n_features = genre_features.shape[1]
        ordered_features = np.zeros((n_items, n_features))
        
        for item_id, matrix_idx in self.item_id_map.items():
            if item_id in item_id_to_idx:
                data_idx = item_id_to_idx[item_id]
                ordered_features[matrix_idx, :] = genre_features[data_idx, :].toarray()
        
        return ordered_features
    
    def _build_user_profiles(
        self, 
        train_data: pd.DataFrame, 
        item_features: np.ndarray
    ) -> np.ndarray:
        """
        Build user profile vectors from their rating history.
        
        Args:
            train_data: Training ratings
            item_features: Item feature matrix
            
        Returns:
            User profile matrix (n_users × n_features)
        """
        n_users = len(self.user_id_map)
        n_features = item_features.shape[1]
        
        user_profiles = np.zeros((n_users, n_features))
        
        for user_id, group in train_data.groupby('user_id'):
            user_idx = self.user_id_map[user_id]
            
            # Weighted average of item features by ratings
            total_weight = 0
            profile = np.zeros(n_features)
            
            for _, row in group.iterrows():
                item_id = row['item_id']
                rating = row['rating']
                
                if item_id in self.item_id_map:
                    item_idx = self.item_id_map[item_id]
                    # Use rating as weight (higher ratings = more influence)
                    profile += rating * item_features[item_idx, :]
                    total_weight += rating
            
            if total_weight > 0:
                user_profiles[user_idx, :] = profile / total_weight
        
        return user_profiles
    
    def predict(self, user_id: int, item_id: int) -> float:
        """
        Predict rating using content-based filtering.
        
        Args:
            user_id: User ID
            item_id: Item ID
            
        Returns:
            Predicted rating (similarity score)
        """
        if user_id not in self.user_id_map or item_id not in self.item_id_map:
            return 2.5  # Neutral rating
        
        user_idx = self.user_id_map[user_id]
        item_idx = self.item_id_map[item_id]
        
        # Calculate similarity between user profile and item
        user_profile = self.user_profiles[user_idx, :]
        item_feature = self.item_features[item_idx, :]
        
        # Cosine similarity
        similarity = np.dot(user_profile, item_feature) / (
            np.linalg.norm(user_profile) * np.linalg.norm(item_feature) + 1e-8
        )
        
        # Convert similarity to rating scale (0-1 → 1-5)
        predicted_rating = 1 + 4 * max(0, similarity)
        
        return predicted_rating
    
    def recommend_similar_items(
        self, 
        item_id: int, 
        n: int = 10
    ) -> List[Tuple[int, float]]:
        """
        Recommend items similar to a given item.
        
        Args:
            item_id: Reference item ID
            n: Number of recommendations
            
        Returns:
            List of (item_id, similarity_score) tuples
        """
        if item_id not in self.item_id_map:
            return []
        
        item_idx = self.item_id_map[item_id]
        
        # Get similarities with all other items
        similarities = self.item_similarity[item_idx, :]
        
        # Get top-N most similar items (excluding itself)
        top_indices = np.argsort(similarities)[::-1][1:n+1]
        
        recommendations = [
            (self.reverse_item_map[idx], similarities[idx])
            for idx in top_indices
        ]
        
        return recommendations


class HybridRecommender(BaseRecommender):
    """Hybrid recommender combining collaborative and content-based filtering."""
    
    def __init__(
        self,
        collaborative_model: BaseRecommender,
        content_model: ContentBasedRecommender,
        alpha: float = 0.5
    ):
        """
        Initialize hybrid recommender.
        
        Args:
            collaborative_model: Collaborative filtering model
            content_model: Content-based model
            alpha: Weight for collaborative filtering (1-alpha for content)
        """
        super().__init__(name="HybridRecommender")
        self.collaborative_model = collaborative_model
        self.content_model = content_model
        self.alpha = alpha
        
    def fit(
        self, 
        train_data: pd.DataFrame,
        movies_data: Optional[pd.DataFrame] = None
    ) -> 'HybridRecommender':
        """
        Fit hybrid recommender.
        
        Args:
            train_data: Training data
            movies_data: Movie metadata (required for content model)
            
        Returns:
            self
        """
        print(f"Fitting {self.name} with alpha={self.alpha}")
        
        # Fit collaborative model
        print("  Fitting collaborative model...")
        self.collaborative_model.fit(train_data)
        
        # Fit content model
        if movies_data is not None:
            print("  Fitting content model...")
            self.content_model.fit(train_data, movies_data)
        
        # Use mappings from collaborative model
        self.user_id_map = self.collaborative_model.user_id_map
        self.item_id_map = self.collaborative_model.item_id_map
        self.reverse_user_map = self.collaborative_model.reverse_user_map
        self.reverse_item_map = self.collaborative_model.reverse_item_map
        
        self.is_fitted = True
        print(f"{self.name} fitted successfully")
        return self
    
    def predict(self, user_id: int, item_id: int) -> float:
        """
        Predict rating using hybrid approach.
        
        Args:
            user_id: User ID
            item_id: Item ID
            
        Returns:
            Predicted rating (weighted combination)
        """
        # Get predictions from both models
        cf_pred = self.collaborative_model.predict(user_id, item_id)
        content_pred = self.content_model.predict(user_id, item_id)
        
        # Weighted combination
        hybrid_pred = self.alpha * cf_pred + (1 - self.alpha) * content_pred
        
        return hybrid_pred
    
    def recommend(
        self,
        user_id: int,
        n: int = 10,
        exclude_seen: bool = True,
        seen_items: Optional[set] = None
    ) -> List[Tuple[int, float]]:
        """
        Generate hybrid recommendations.
        
        Args:
            user_id: User ID
            n: Number of recommendations
            exclude_seen: Whether to exclude seen items
            seen_items: Set of seen item IDs
            
        Returns:
            List of (item_id, score) tuples
        """
        # Get recommendations from both models
        cf_recs = self.collaborative_model.recommend(
            user_id, n=n*2, exclude_seen=exclude_seen, seen_items=seen_items
        )
        content_recs = self.content_model.recommend(
            user_id, n=n*2, exclude_seen=exclude_seen, seen_items=seen_items
        )
        
        # Combine scores
        combined_scores = {}
        
        for item_id, score in cf_recs:
            combined_scores[item_id] = self.alpha * score
        
        for item_id, score in content_recs:
            if item_id in combined_scores:
                combined_scores[item_id] += (1 - self.alpha) * score
            else:
                combined_scores[item_id] = (1 - self.alpha) * score
        
        # Sort and return top-N
        sorted_items = sorted(
            combined_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        return sorted_items[:n]


if __name__ == "__main__":
    # Example usage
    from src.data_loader import MovieLensLoader
    from src.preprocess import prepare_data_for_training
    from src.recommenders.collaborative import ItemBasedCF
    
    # Load data
    loader = MovieLensLoader()
    ratings = loader.load_ratings()
    movies = loader.load_movies()
    
    # Prepare data
    train, test, _ = prepare_data_for_training(
        ratings,
        test_size=0.2,
        split_method='user_based'
    )
    
    print("\n" + "="*60)
    print("Testing Content-Based Recommender")
    print("="*60)
    
    # Content-based
    print("\n1. Content-Based Recommender")
    content_rec = ContentBasedRecommender()
    content_rec.fit(train, movies)
    pred = content_rec.predict(user_id=1, item_id=100)
    print(f"Prediction for user=1, item=100: {pred:.2f}")
    
    # Similar items
    print("\nSimilar movies to item 1:")
    similar = content_rec.recommend_similar_items(item_id=1, n=5)
    for item_id, score in similar:
        movie_title = movies[movies['item_id'] == item_id]['title'].values[0]
        print(f"  {movie_title}: {score:.3f}")
    
    # Hybrid
    print("\n2. Hybrid Recommender")
    item_cf = ItemBasedCF(k=50)
    hybrid_rec = HybridRecommender(
        collaborative_model=item_cf,
        content_model=content_rec,
        alpha=0.7  # 70% collaborative, 30% content
    )
    hybrid_rec.fit(train, movies)
    pred = hybrid_rec.predict(user_id=1, item_id=100)
    print(f"Prediction for user=1, item=100: {pred:.2f}")