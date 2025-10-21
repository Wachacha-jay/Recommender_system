"""
Data loader for MovieLens dataset.
Handles downloading, loading, and basic data structures.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile
from typing import Tuple, Dict
import warnings
warnings.filterwarnings('ignore')


class MovieLensLoader:
    """Loader for MovieLens 100K dataset."""
    
    DATASET_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
    
    def __init__(self, data_dir: str = "data/raw"):
        """
        Initialize the data loader.
        
        Args:
            data_dir: Directory to store raw data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_path = self.data_dir / "ml-100k"
        
    def download_data(self) -> None:
        """Download MovieLens 100K dataset if not already present."""
        zip_path = self.data_dir / "ml-100k.zip"
        
        if self.dataset_path.exists():
            print(f"Dataset already exists at {self.dataset_path}")
            return
            
        print(f"Downloading MovieLens 100K dataset...")
        urlretrieve(self.DATASET_URL, zip_path)
        
        print("Extracting dataset...")
        with ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.data_dir)
        
        # Clean up zip file
        zip_path.unlink()
        print(f"Dataset downloaded and extracted to {self.dataset_path}")
    
    def load_ratings(self) -> pd.DataFrame:
        """
        Load ratings data.
        
        Returns:
            DataFrame with columns: user_id, item_id, rating, timestamp
        """
        ratings_path = self.dataset_path / "u.data"
        
        if not ratings_path.exists():
            raise FileNotFoundError(
                f"Ratings file not found. Run download_data() first."
            )
        
        ratings = pd.read_csv(
            ratings_path,
            sep='\t',
            names=['user_id', 'item_id', 'rating', 'timestamp'],
            engine='python'
        )
        
        print(f"Loaded {len(ratings):,} ratings")
        print(f"Users: {ratings['user_id'].nunique():,}")
        print(f"Items: {ratings['item_id'].nunique():,}")
        print(f"Sparsity: {1 - len(ratings) / (ratings['user_id'].nunique() * ratings['item_id'].nunique()):.4f}")
        
        return ratings
    
    def load_movies(self) -> pd.DataFrame:
        """
        Load movie metadata.
        
        Returns:
            DataFrame with movie information including title, genres
        """
        movies_path = self.dataset_path / "u.item"
        
        if not movies_path.exists():
            raise FileNotFoundError(
                f"Movies file not found. Run download_data() first."
            )
        
        # Column names for u.item file
        columns = [
            'item_id', 'title', 'release_date', 'video_release_date',
            'imdb_url', 'unknown', 'Action', 'Adventure', 'Animation',
            'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
            'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
            'Thriller', 'War', 'Western'
        ]
        
        movies = pd.read_csv(
            movies_path,
            sep='|',
            names=columns,
            encoding='latin-1',
            engine='python'
        )
        
        # Create genres list for each movie
        genre_columns = columns[5:]  # All genre columns
        movies['genres'] = movies[genre_columns].apply(
            lambda row: [genre for genre, val in row.items() if val == 1],
            axis=1
        )
        
        # Keep only essential columns
        movies = movies[['item_id', 'title', 'release_date', 'genres']]
        
        print(f"Loaded {len(movies):,} movies")
        
        return movies
    
    def load_users(self) -> pd.DataFrame:
        """
        Load user demographic data.
        
        Returns:
            DataFrame with user demographics
        """
        users_path = self.dataset_path / "u.user"
        
        if not users_path.exists():
            raise FileNotFoundError(
                f"Users file not found. Run download_data() first."
            )
        
        users = pd.read_csv(
            users_path,
            sep='|',
            names=['user_id', 'age', 'gender', 'occupation', 'zip_code'],
            engine='python'
        )
        
        print(f"Loaded {len(users):,} users")
        
        return users
    
    def create_interaction_matrix(
        self, 
        ratings: pd.DataFrame,
        binary: bool = False
    ) -> Tuple[np.ndarray, Dict[int, int], Dict[int, int]]:
        """
        Create user-item interaction matrix.
        
        Args:
            ratings: DataFrame with user_id, item_id, rating
            binary: If True, convert to binary (1 if rated, 0 otherwise)
        
        Returns:
            Tuple of (interaction_matrix, user_id_map, item_id_map)
        """
        # Create mappings from original IDs to matrix indices
        unique_users = sorted(ratings['user_id'].unique())
        unique_items = sorted(ratings['item_id'].unique())
        
        user_id_map = {uid: idx for idx, uid in enumerate(unique_users)}
        item_id_map = {iid: idx for idx, iid in enumerate(unique_items)}
        
        n_users = len(unique_users)
        n_items = len(unique_items)
        
        # Create sparse matrix
        matrix = np.zeros((n_users, n_items))
        
        for _, row in ratings.iterrows():
            user_idx = user_id_map[row['user_id']]
            item_idx = item_id_map[row['item_id']]
            matrix[user_idx, item_idx] = 1.0 if binary else row['rating']
        
        print(f"Created interaction matrix: {matrix.shape}")
        print(f"Density: {np.count_nonzero(matrix) / matrix.size:.4f}")
        
        return matrix, user_id_map, item_id_map


def download_movielens():
    """Convenience function to download MovieLens data."""
    loader = MovieLensLoader()
    loader.download_data()
    print("\nData downloaded successfully!")
    print("You can now load it using:")
    print("  from src.data_loader import MovieLensLoader")
    print("  loader = MovieLensLoader()")
    print("  ratings = loader.load_ratings()")
    print("  movies = loader.load_movies()")


if __name__ == "__main__":
    # Example usage
    loader = MovieLensLoader()
    loader.download_data()
    
    ratings = loader.load_ratings()
    movies = loader.load_movies()
    users = loader.load_users()
    
    print("\n" + "="*50)
    print("Sample ratings:")
    print(ratings.head())
    
    print("\n" + "="*50)
    print("Sample movies:")
    print(movies.head())
    
    print("\n" + "="*50)
    print("Sample users:")
    print(users.head())