"""
Utility functions for recommendation systems.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


def plot_rating_distribution(ratings: pd.DataFrame) -> None:
    """
    Plot rating distribution.
    
    Args:
        ratings: DataFrame with ratings
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Rating distribution
    ratings['rating'].value_counts().sort_index().plot(
        kind='bar', ax=axes[0], color='skyblue'
    )
    axes[0].set_title('Rating Distribution')
    axes[0].set_xlabel('Rating')
    axes[0].set_ylabel('Count')
    
    # Ratings per user
    user_counts = ratings.groupby('user_id').size()
    axes[1].hist(user_counts, bins=50, color='coral', edgecolor='black')
    axes[1].set_title('Ratings per User Distribution')
    axes[1].set_xlabel('Number of Ratings')
    axes[1].set_ylabel('Number of Users')
    axes[1].axvline(user_counts.median(), color='red', linestyle='--', 
                    label=f'Median: {user_counts.median():.0f}')
    axes[1].legend()
    
    plt.tight_layout()
    plt.show()


def plot_sparsity_heatmap(
    ratings: pd.DataFrame,
    sample_users: int = 50,
    sample_items: int = 50
) -> None:
    """
    Plot sparsity heatmap for a sample of users and items.
    
    Args:
        ratings: DataFrame with ratings
        sample_users: Number of users to sample
        sample_items: Number of items to sample
    """
    # Create user-item matrix
    matrix = ratings.pivot_table(
        index='user_id',
        columns='item_id',
        values='rating'
    )
    
    # Sample users and items
    sampled_users = np.random.choice(matrix.index, size=min(sample_users, len(matrix)), replace=False)
    sampled_items = np.random.choice(matrix.columns, size=min(sample_items, len(matrix.columns)), replace=False)
    
    sample_matrix = matrix.loc[sampled_users, sampled_items]
    
    # Plot
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        sample_matrix.notna().astype(int),
        cmap='YlOrRd',
        cbar_kws={'label': 'Has Rating'},
        xticklabels=False,
        yticklabels=False
    )
    plt.title(f'User-Item Interaction Matrix (Sample)\nSparsity: {1 - sample_matrix.notna().sum().sum() / sample_matrix.size:.4f}')
    plt.xlabel('Items')
    plt.ylabel('Users')
    plt.tight_layout()
    plt.show()


def plot_model_comparison(
    results_df: pd.DataFrame,
    metrics: List[str] = None
) -> None:
    """
    Plot comparison of multiple models.
    
    Args:
        results_df: DataFrame with models as rows and metrics as columns
        metrics: List of metrics to plot (None = all)
    """
    if metrics is None:
        metrics = results_df.columns.tolist()
    
    # Filter metrics
    plot_data = results_df[metrics]
    
    # Create subplots
    n_metrics = len(metrics)
    n_cols = min(3, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    if n_metrics == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_rows > 1 else axes
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx] if n_metrics > 1 else axes[0]
        
        plot_data[metric].plot(
            kind='bar',
            ax=ax,
            color='steelblue',
            edgecolor='black'
        )
        ax.set_title(metric, fontsize=12, fontweight='bold')
        ax.set_ylabel('Score')
        ax.set_xlabel('Model')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_metrics, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.show()


def plot_precision_recall_curve(
    results: Dict[str, Dict[int, Tuple[float, float]]],
    model_names: List[str] = None
) -> None:
    """
    Plot precision-recall curves for different K values.
    
    Args:
        results: Dict of {model_name: {k: (precision, recall)}}
        model_names: List of model names to plot (None = all)
    """
    plt.figure(figsize=(10, 6))
    
    if model_names is None:
        model_names = list(results.keys())
    
    for model_name in model_names:
        if model_name in results:
            k_values = sorted(results[model_name].keys())
            precisions = [results[model_name][k][0] for k in k_values]
            recalls = [results[model_name][k][1] for k in k_values]
            
            plt.plot(recalls, precisions, marker='o', label=model_name, linewidth=2)
    
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curves', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def analyze_user_behavior(ratings: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze user rating behavior.
    
    Args:
        ratings: DataFrame with ratings
        
    Returns:
        DataFrame with user statistics
    """
    user_stats = ratings.groupby('user_id').agg({
        'rating': ['count', 'mean', 'std', 'min', 'max'],
        'item_id': 'nunique'
    })
    
    user_stats.columns = [
        'num_ratings', 'avg_rating', 'std_rating', 
        'min_rating', 'max_rating', 'num_unique_items'
    ]
    
    # Add rating variance category
    user_stats['rating_behavior'] = pd.cut(
        user_stats['std_rating'],
        bins=[0, 0.5, 1.0, 2.0, 5.0],
        labels=['Very Consistent', 'Consistent', 'Varied', 'Very Varied']
    )
    
    print("User Behavior Summary:")
    print("="*60)
    print(f"Total users: {len(user_stats):,}")
    print(f"Avg ratings per user: {user_stats['num_ratings'].mean():.1f}")
    print(f"Avg rating: {user_stats['avg_rating'].mean():.2f}")
    print(f"\nRating Behavior Distribution:")
    print(user_stats['rating_behavior'].value_counts())
    
    return user_stats


def analyze_item_popularity(ratings: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze item popularity.
    
    Args:
        ratings: DataFrame with ratings
        
    Returns:
        DataFrame with item statistics
    """
    item_stats = ratings.groupby('item_id').agg({
        'rating': ['count', 'mean', 'std'],
        'user_id': 'nunique'
    })
    
    item_stats.columns = ['num_ratings', 'avg_rating', 'std_rating', 'num_unique_users']
    
    # Add popularity category
    item_stats['popularity'] = pd.qcut(
        item_stats['num_ratings'],
        q=4,
        labels=['Niche', 'Moderate', 'Popular', 'Very Popular']
    )
    
    print("Item Popularity Summary:")
    print("="*60)
    print(f"Total items: {len(item_stats):,}")
    print(f"Avg ratings per item: {item_stats['num_ratings'].mean():.1f}")
    print(f"\nPopularity Distribution:")
    print(item_stats['popularity'].value_counts())
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Popularity distribution
    item_stats['num_ratings'].hist(bins=50, ax=axes[0], color='lightcoral', edgecolor='black')
    axes[0].set_title('Item Popularity Distribution')
    axes[0].set_xlabel('Number of Ratings')
    axes[0].set_ylabel('Number of Items')
    axes[0].set_yscale('log')
    
    # Average rating vs popularity
    axes[1].scatter(
        item_stats['num_ratings'],
        item_stats['avg_rating'],
        alpha=0.5,
        color='steelblue'
    )
    axes[1].set_title('Average Rating vs Popularity')
    axes[1].set_xlabel('Number of Ratings')
    axes[1].set_ylabel('Average Rating')
    axes[1].set_xscale('log')
    
    plt.tight_layout()
    plt.show()
    
    return item_stats


def create_train_test_timeline(ratings: pd.DataFrame, train: pd.DataFrame, test: pd.DataFrame) -> None:
    """
    Visualize temporal split of train/test data.
    
    Args:
        ratings: Full ratings DataFrame
        train: Train DataFrame
        test: Test DataFrame
    """
    if 'timestamp' not in ratings.columns:
        print("Timestamp column not found")
        return
    
    # Convert timestamps to dates
    ratings['date'] = pd.to_datetime(ratings['timestamp'], unit='s')
    train['date'] = pd.to_datetime(train['timestamp'], unit='s')
    test['date'] = pd.to_datetime(test['timestamp'], unit='s')
    
    # Count ratings per day
    train_counts = train.groupby(train['date'].dt.date).size()
    test_counts = test.groupby(test['date'].dt.date).size()
    
    plt.figure(figsize=(14, 6))
    plt.plot(train_counts.index, train_counts.values, label='Train', alpha=0.7, linewidth=2)
    plt.plot(test_counts.index, test_counts.values, label='Test', alpha=0.7, linewidth=2)
    plt.xlabel('Date')
    plt.ylabel('Number of Ratings')
    plt.title('Train/Test Split Timeline')
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def save_results(results: Dict, filepath: str) -> None:
    """
    Save evaluation results to file.
    
    Args:
        results: Dictionary of results
        filepath: Path to save file
    """
    import json
    
    # Convert numpy types to Python types
    cleaned_results = {}
    for key, value in results.items():
        if isinstance(value, (np.integer, np.floating)):
            cleaned_results[key] = float(value)
        elif isinstance(value, dict):
            cleaned_results[key] = {k: float(v) if isinstance(v, (np.integer, np.floating)) else v 
                                   for k, v in value.items()}
        else:
            cleaned_results[key] = value
    
    with open(filepath, 'w') as f:
        json.dump(cleaned_results, f, indent=2)
    
    print(f"Results saved to {filepath}")


if __name__ == "__main__":
    # Example usage
    from src.data_loader import MovieLensLoader
    
    loader = MovieLensLoader()
    ratings = loader.load_ratings()
    
    # Plot distributions
    plot_rating_distribution(ratings)
    
    # Analyze user behavior
    user_stats = analyze_user_behavior(ratings)
    
    # Analyze item popularity
    item_stats = analyze_item_popularity(ratings)
    
    # Plot sparsity
    plot_sparsity_heatmap(ratings, sample_users=100, sample_items=100)