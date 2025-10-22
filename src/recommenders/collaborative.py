"""
Collaborative filtering recommendation algorithms.
Includes user-based, item-based, and matrix factorization methods.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
import sys
sys.path.append('..')
from src.recommenders.base import BaseRecommender


class UserBasedCF(BaseRecommender):
    """User-based collaborative filtering."""
    
    def __init__(
        self, 
        k: int = 50,
        similarity: str = 'cosine',
        min_support: int = 1
    ):
        """
        Initialize user-based CF.
        
        Args:
            k: Number of similar users to consider
            similarity: Similarity metric ('cosine' or 'pearson')
            min_support: Minimum number of common ratings
        """
        super().__init__(name="UserBasedCF")
        self.k = k
        self.similarity_metric = similarity
        self.min_support = min_support
        self.user_similarity = None
        self.user_item_matrix = None
        self.user_means = None
        
    def fit(self, train_data: pd.DataFrame) -> 'UserBasedCF':
        """
        Fit user-based CF model.
        
        Args:
            train_data: Training data
            
        Returns:
            self
        """
        self._create_mappings(train_data)
        
        n_users = len(self.user_id_map)
        n_items = len(self.item_id_map)
        
        # Create user-item matrix
        self.user_item_matrix = np.zeros((n_users, n_items))
        
        for _, row in train_data.iterrows():
            user_idx = self.user_id_map[row['user_id']]
            item_idx = self.item_id_map[row['item_id']]
            self.user_item_matrix[user_idx, item_idx] = row['rating']
        
        # Calculate user means (for mean-centering)
        self.user_means = np.zeros(n_users)
        for i in range(n_users):
            user_ratings = self.user_item_matrix[i, :]
            rated_items = user_ratings > 0
            if rated_items.sum() > 0:
                self.user_means[i] = user_ratings[rated_items].mean()
        
        # Mean-center the matrix
        centered_matrix = self.user_item_matrix.copy()
        for i in range(n_users):
            rated_items = self.user_item_matrix[i, :] > 0
            centered_matrix[i, rated_items] -= self.user_means[i]
        
        # Calculate user similarity
        if self.similarity_metric == 'cosine':
            # Replace zeros with small value to avoid division by zero
            centered_matrix_filled = np.where(
                self.user_item_matrix > 0, centered_matrix, 0
            )
            self.user_similarity = cosine_similarity(centered_matrix_filled)
            
        elif self.similarity_metric == 'pearson':
            # Pearson correlation
            self.user_similarity = np.corrcoef(centered_matrix)
            # Handle NaN values
            self.user_similarity = np.nan_to_num(self.user_similarity)
        
        # Set self-similarity to 0
        np.fill_diagonal(self.user_similarity, 0)
        
        self.is_fitted = True
        print(f"{self.name} fitted with k={self.k}, similarity='{self.similarity_metric}'")
        return self
    
    def predict(self, user_id: int, item_id: int) -> float:
        """
        Predict rating using user-based CF.
        
        Args:
            user_id: User ID
            item_id: Item ID
            
        Returns:
            Predicted rating
        """
        if user_id not in self.user_id_map or item_id not in self.item_id_map:
            # Return global mean for unknown users/items
            return self.user_means.mean()
        
        user_idx = self.user_id_map[user_id]
        item_idx = self.item_id_map[item_id]
        
        # Get similar users who rated this item
        similar_users = self.user_similarity[user_idx, :]
        rated_mask = self.user_item_matrix[:, item_idx] > 0
        
        # Get top-k similar users who rated the item
        valid_users = similar_users * rated_mask
        top_k_indices = np.argsort(valid_users)[-self.k:]
        top_k_indices = top_k_indices[valid_users[top_k_indices] > 0]
        
        if len(top_k_indices) == 0:
            return self.user_means[user_idx]
        
        # Weighted average of ratings
        similarities = self.user_similarity[user_idx, top_k_indices]
        ratings = self.user_item_matrix[top_k_indices, item_idx]
        neighbor_means = self.user_means[top_k_indices]
        
        # Prediction = user_mean + weighted_avg(neighbor_rating - neighbor_mean)
        numerator = np.sum(similarities * (ratings - neighbor_means))
        denominator = np.sum(np.abs(similarities))
        
        if denominator == 0:
            return self.user_means[user_idx]
        
        prediction = self.user_means[user_idx] + (numerator / denominator)
        return np.clip(prediction, 1, 5)


class ItemBasedCF(BaseRecommender):
    """Item-based collaborative filtering."""
    
    def __init__(
        self, 
        k: int = 50,
        similarity: str = 'cosine',
        min_support: int = 1
    ):
        """
        Initialize item-based CF.
        
        Args:
            k: Number of similar items to consider
            similarity: Similarity metric ('cosine' or 'pearson')
            min_support: Minimum number of common users
        """
        super().__init__(name="ItemBasedCF")
        self.k = k
        self.similarity_metric = similarity
        self.min_support = min_support
        self.item_similarity = None
        self.user_item_matrix = None
        self.item_means = None
        
    def fit(self, train_data: pd.DataFrame) -> 'ItemBasedCF':
        """
        Fit item-based CF model.
        
        Args:
            train_data: Training data
            
        Returns:
            self
        """
        self._create_mappings(train_data)
        
        n_users = len(self.user_id_map)
        n_items = len(self.item_id_map)
        
        # Create user-item matrix
        self.user_item_matrix = np.zeros((n_users, n_items))
        
        for _, row in train_data.iterrows():
            user_idx = self.user_id_map[row['user_id']]
            item_idx = self.item_id_map[row['item_id']]
            self.user_item_matrix[user_idx, item_idx] = row['rating']
        
        # Calculate item means
        self.item_means = np.zeros(n_items)
        for j in range(n_items):
            item_ratings = self.user_item_matrix[:, j]
            rated_users = item_ratings > 0
            if rated_users.sum() > 0:
                self.item_means[j] = item_ratings[rated_users].mean()
        
        # Mean-center by item
        centered_matrix = self.user_item_matrix.copy()
        for j in range(n_items):
            rated_users = self.user_item_matrix[:, j] > 0
            centered_matrix[rated_users, j] -= self.item_means[j]
        
        # Calculate item similarity (transpose to get items as rows)
        if self.similarity_metric == 'cosine':
            centered_matrix_filled = np.where(
                self.user_item_matrix > 0, centered_matrix, 0
            )
            self.item_similarity = cosine_similarity(centered_matrix_filled.T)
            
        elif self.similarity_metric == 'pearson':
            self.item_similarity = np.corrcoef(centered_matrix.T)
            self.item_similarity = np.nan_to_num(self.item_similarity)
        
        # Set self-similarity to 0
        np.fill_diagonal(self.item_similarity, 0)
        
        self.is_fitted = True
        print(f"{self.name} fitted with k={self.k}, similarity='{self.similarity_metric}'")
        return self
    
    def predict(self, user_id: int, item_id: int) -> float:
        """
        Predict rating using item-based CF.
        
        Args:
            user_id: User ID
            item_id: Item ID
            
        Returns:
            Predicted rating
        """
        if user_id not in self.user_id_map or item_id not in self.item_id_map:
            return self.item_means.mean()
        
        user_idx = self.user_id_map[user_id]
        item_idx = self.item_id_map[item_id]
        
        # Get items user has rated
        user_ratings = self.user_item_matrix[user_idx, :]
        rated_items = user_ratings > 0
        
        if not rated_items.any():
            return self.item_means[item_idx]
        
        # Get similar items that user has rated
        similar_items = self.item_similarity[item_idx, :]
        valid_items = similar_items * rated_items
        
        # Get top-k similar items
        top_k_indices = np.argsort(valid_items)[-self.k:]
        top_k_indices = top_k_indices[valid_items[top_k_indices] > 0]
        
        if len(top_k_indices) == 0:
            return self.item_means[item_idx]
        
        # Weighted average
        similarities = self.item_similarity[item_idx, top_k_indices]
        ratings = self.user_item_matrix[user_idx, top_k_indices]
        
        numerator = np.sum(similarities * ratings)
        denominator = np.sum(np.abs(similarities))
        
        if denominator == 0:
            return self.item_means[item_idx]
        
        prediction = numerator / denominator
        return np.clip(prediction, 1, 5)


class MatrixFactorizationSVD(BaseRecommender):
    """Matrix Factorization using Singular Value Decomposition."""
    
    def __init__(self, n_factors: int = 50, random_state: int = 42):
        """
        Initialize SVD-based matrix factorization.
        
        Args:
            n_factors: Number of latent factors
            random_state: Random seed
        """
        super().__init__(name="MatrixFactorizationSVD")
        self.n_factors = n_factors
        self.random_state = random_state
        self.user_factors = None
        self.item_factors = None
        self.global_mean = None
        
    def fit(self, train_data: pd.DataFrame) -> 'MatrixFactorizationSVD':
        """
        Fit SVD model.
        
        Args:
            train_data: Training data
            
        Returns:
            self
        """
        self._create_mappings(train_data)
        
        n_users = len(self.user_id_map)
        n_items = len(self.item_id_map)
        
        # Create sparse user-item matrix
        rows, cols, ratings = [], [], []
        for _, row in train_data.iterrows():
            user_idx = self.user_id_map[row['user_id']]
            item_idx = self.item_id_map[row['item_id']]
            rows.append(user_idx)
            cols.append(item_idx)
            ratings.append(row['rating'])
        
        sparse_matrix = csr_matrix(
            (ratings, (rows, cols)), 
            shape=(n_users, n_items)
        )
        
        # Calculate global mean
        self.global_mean = np.mean(ratings)
        
        # Perform SVD
        # Use min of (n_factors, min(matrix dimensions) - 1)
        k = min(self.n_factors, min(n_users, n_items) - 1)
        
        U, sigma, Vt = svds(sparse_matrix.astype(np.float64), k=k)
        
        # Store factors
        self.user_factors = U
        self.item_factors = Vt.T
        self.sigma = sigma
        
        self.is_fitted = True
        print(f"{self.name} fitted with {k} latent factors")
        return self
    
    def predict(self, user_id: int, item_id: int) -> float:
        """
        Predict rating using matrix factorization.
        
        Args:
            user_id: User ID
            item_id: Item ID
            
        Returns:
            Predicted rating
        """
        if user_id not in self.user_id_map or item_id not in self.item_id_map:
            return self.global_mean
        
        user_idx = self.user_id_map[user_id]
        item_idx = self.item_id_map[item_id]
        
        # Prediction = user_factors · sigma · item_factors
        prediction = np.dot(
            np.dot(self.user_factors[user_idx, :], np.diag(self.sigma)),
            self.item_factors[item_idx, :]
        )
        
        return np.clip(prediction, 1, 5)


class AlternatingLeastSquares(BaseRecommender):
    """Matrix Factorization using Alternating Least Squares."""
    
    def __init__(
        self,
        n_factors: int = 50,
        regularization: float = 0.01,
        iterations: int = 20,
        random_state: int = 42
    ):
        """
        Initialize ALS.
        
        Args:
            n_factors: Number of latent factors
            regularization: L2 regularization parameter
            iterations: Number of iterations
            random_state: Random seed
        """
        super().__init__(name="AlternatingLeastSquares")
        self.n_factors = n_factors
        self.regularization = regularization
        self.iterations = iterations
        self.random_state = random_state
        self.user_factors = None
        self.item_factors = None
        self.global_mean = None
        
    def fit(self, train_data: pd.DataFrame) -> 'AlternatingLeastSquares':
        """
        Fit ALS model.
        
        Args:
            train_data: Training data
            
        Returns:
            self
        """
        self._create_mappings(train_data)
        
        n_users = len(self.user_id_map)
        n_items = len(self.item_id_map)
        
        # Create user-item matrix
        user_item_matrix = np.zeros((n_users, n_items))
        for _, row in train_data.iterrows():
            user_idx = self.user_id_map[row['user_id']]
            item_idx = self.item_id_map[row['item_id']]
            user_item_matrix[user_idx, item_idx] = row['rating']
        
        self.global_mean = train_data['rating'].mean()
        
        # Initialize factors randomly
        np.random.seed(self.random_state)
        self.user_factors = np.random.normal(0, 0.1, (n_users, self.n_factors))
        self.item_factors = np.random.normal(0, 0.1, (n_items, self.n_factors))
        
        # ALS iterations
        for iteration in range(self.iterations):
            # Fix item factors, update user factors
            for u in range(n_users):
                rated_items = user_item_matrix[u, :] > 0
                if not rated_items.any():
                    continue
                
                X = self.item_factors[rated_items, :]
                y = user_item_matrix[u, rated_items]
                
                # Solve: (X^T X + λI) p_u = X^T y
                XtX = X.T @ X
                Xty = X.T @ y
                
                reg_matrix = self.regularization * np.eye(self.n_factors)
                self.user_factors[u, :] = np.linalg.solve(XtX + reg_matrix, Xty)
            
            # Fix user factors, update item factors
            for i in range(n_items):
                rated_users = user_item_matrix[:, i] > 0
                if not rated_users.any():
                    continue
                
                X = self.user_factors[rated_users, :]
                y = user_item_matrix[rated_users, i]
                
                XtX = X.T @ X
                Xty = X.T @ y
                
                reg_matrix = self.regularization * np.eye(self.n_factors)
                self.item_factors[i, :] = np.linalg.solve(XtX + reg_matrix, Xty)
            
            if (iteration + 1) % 5 == 0:
                # Calculate training error
                predictions = self.user_factors @ self.item_factors.T
                mask = user_item_matrix > 0
                error = np.sqrt(np.mean((user_item_matrix[mask] - predictions[mask]) ** 2))
                print(f"Iteration {iteration + 1}/{self.iterations}, RMSE: {error:.4f}")
        
        self.is_fitted = True
        print(f"{self.name} training complete")
        return self
    
    def predict(self, user_id: int, item_id: int) -> float:
        """
        Predict rating using ALS.
        
        Args:
            user_id: User ID
            item_id: Item ID
            
        Returns:
            Predicted rating
        """
        if user_id not in self.user_id_map or item_id not in self.item_id_map:
            return self.global_mean
        
        user_idx = self.user_id_map[user_id]
        item_idx = self.item_id_map[item_id]
        
        prediction = np.dot(self.user_factors[user_idx, :], self.item_factors[item_idx, :])
        return np.clip(prediction, 1, 5)


if __name__ == "__main__":
    # Example usage
    from src.data_loader import MovieLensLoader
    from src.preprocess import prepare_data_for_training
    
    # Load data
    loader = MovieLensLoader()
    ratings = loader.load_ratings()
    
    # Prepare data
    train, test, _ = prepare_data_for_training(
        ratings, 
        test_size=0.2,
        split_method='user_based'
    )
    
    print("\n" + "="*60)
    print("Testing Collaborative Filtering Methods")
    print("="*60)
    
    # User-based CF
    print("\n1. User-Based CF")
    user_cf = UserBasedCF(k=50)
    user_cf.fit(train)
    pred = user_cf.predict(user_id=1, item_id=100)
    print(f"Prediction for user=1, item=100: {pred:.2f}")
    
    # Item-based CF
    print("\n2. Item-Based CF")
    item_cf = ItemBasedCF(k=50)
    item_cf.fit(train)
    pred = item_cf.predict(user_id=1, item_id=100)
    print(f"Prediction for user=1, item=100: {pred:.2f}")
    
    # SVD
    print("\n3. Matrix Factorization (SVD)")
    svd_model = MatrixFactorizationSVD(n_factors=50)
    svd_model.fit(train)
    pred = svd_model.predict(user_id=1, item_id=100)
    print(f"Prediction for user=1, item=100: {pred:.2f}")
    
    # ALS
    print("\n4. Alternating Least Squares")
    als_model = AlternatingLeastSquares(n_factors=20, iterations=10)
    als_model.fit(train)
    pred = als_model.predict(user_id=1, item_id=100)
    print(f"Prediction for user=1, item=100: {pred:.2f}")