"""
Evaluation metrics for recommendation systems.
Includes ranking metrics, coverage, diversity, and more.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Set
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


class RecommenderEvaluator:
    """Comprehensive evaluation for recommender systems."""
    
    def __init__(self, k_values: List[int] = [5, 10, 20]):
        """
        Initialize evaluator.
        
        Args:
            k_values: List of K values for top-K metrics
        """
        self.k_values = k_values
        
    def precision_at_k(
        self,
        recommended: List[int],
        relevant: Set[int],
        k: int
    ) -> float:
        """
        Calculate Precision@K.
        
        Args:
            recommended: List of recommended item IDs (ranked)
            relevant: Set of relevant (ground truth) item IDs
            k: Top-K cutoff
            
        Returns:
            Precision@K score
        """
        if k == 0 or len(recommended) == 0:
            return 0.0
        
        top_k = recommended[:k]
        num_relevant = len([item for item in top_k if item in relevant])
        
        return num_relevant / k
    
    def recall_at_k(
        self,
        recommended: List[int],
        relevant: Set[int],
        k: int
    ) -> float:
        """
        Calculate Recall@K.
        
        Args:
            recommended: List of recommended item IDs (ranked)
            relevant: Set of relevant (ground truth) item IDs
            k: Top-K cutoff
            
        Returns:
            Recall@K score
        """
        if len(relevant) == 0:
            return 0.0
        
        top_k = recommended[:k]
        num_relevant = len([item for item in top_k if item in relevant])
        
        return num_relevant / len(relevant)
    
    def f1_at_k(
        self,
        recommended: List[int],
        relevant: Set[int],
        k: int
    ) -> float:
        """
        Calculate F1@K score.
        
        Args:
            recommended: List of recommended item IDs
            relevant: Set of relevant item IDs
            k: Top-K cutoff
            
        Returns:
            F1@K score
        """
        precision = self.precision_at_k(recommended, relevant, k)
        recall = self.recall_at_k(recommended, relevant, k)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    def average_precision(
        self,
        recommended: List[int],
        relevant: Set[int]
    ) -> float:
        """
        Calculate Average Precision.
        
        Args:
            recommended: List of recommended item IDs (ranked)
            relevant: Set of relevant item IDs
            
        Returns:
            Average Precision score
        """
        if len(relevant) == 0:
            return 0.0
        
        score = 0.0
        num_hits = 0.0
        
        for k, item in enumerate(recommended, start=1):
            if item in relevant:
                num_hits += 1.0
                precision_at_k = num_hits / k
                score += precision_at_k
        
        return score / len(relevant)
    
    def mean_average_precision(
        self,
        recommendations: Dict[int, List[int]],
        ground_truth: Dict[int, Set[int]]
    ) -> float:
        """
        Calculate Mean Average Precision (MAP).
        
        Args:
            recommendations: Dict mapping user_id to recommended items
            ground_truth: Dict mapping user_id to relevant items
            
        Returns:
            MAP score
        """
        ap_scores = []
        
        for user_id in recommendations:
            if user_id in ground_truth:
                ap = self.average_precision(
                    recommendations[user_id],
                    ground_truth[user_id]
                )
                ap_scores.append(ap)
        
        return np.mean(ap_scores) if ap_scores else 0.0
    
    def dcg_at_k(
        self,
        recommended: List[int],
        relevant: Set[int],
        k: int
    ) -> float:
        """
        Calculate Discounted Cumulative Gain@K.
        
        Args:
            recommended: List of recommended item IDs (ranked)
            relevant: Set of relevant item IDs
            k: Top-K cutoff
            
        Returns:
            DCG@K score
        """
        dcg = 0.0
        for i, item in enumerate(recommended[:k], start=1):
            if item in relevant:
                # rel = 1 if relevant, 0 otherwise
                dcg += 1.0 / np.log2(i + 1)
        
        return dcg
    
    def ndcg_at_k(
        self,
        recommended: List[int],
        relevant: Set[int],
        k: int
    ) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain@K.
        
        Args:
            recommended: List of recommended item IDs (ranked)
            relevant: Set of relevant item IDs
            k: Top-K cutoff
            
        Returns:
            NDCG@K score
        """
        dcg = self.dcg_at_k(recommended, relevant, k)
        
        # Ideal DCG (all relevant items at top)
        ideal_relevant = list(relevant)[:k]
        idcg = sum(1.0 / np.log2(i + 2) for i in range(len(ideal_relevant)))
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    def hit_rate_at_k(
        self,
        recommended: List[int],
        relevant: Set[int],
        k: int
    ) -> float:
        """
        Calculate Hit Rate@K (binary: did we hit at least one relevant item?).
        
        Args:
            recommended: List of recommended item IDs
            relevant: Set of relevant item IDs
            k: Top-K cutoff
            
        Returns:
            Hit rate (1.0 or 0.0)
        """
        top_k = recommended[:k]
        return 1.0 if any(item in relevant for item in top_k) else 0.0
    
    def coverage(
        self,
        recommendations: Dict[int, List[int]],
        all_items: Set[int]
    ) -> float:
        """
        Calculate catalog coverage.
        
        Args:
            recommendations: Dict mapping user_id to recommended items
            all_items: Set of all available items
            
        Returns:
            Coverage score (0-1)
        """
        recommended_items = set()
        for rec_list in recommendations.values():
            recommended_items.update(rec_list)
        
        return len(recommended_items) / len(all_items)
    
    def diversity(
        self,
        recommendations: Dict[int, List[int]],
        item_similarity: np.ndarray,
        item_id_map: Dict[int, int]
    ) -> float:
        """
        Calculate average intra-list diversity.
        
        Args:
            recommendations: Dict mapping user_id to recommended items
            item_similarity: Item similarity matrix
            item_id_map: Mapping from item_id to matrix index
            
        Returns:
            Average diversity score
        """
        diversity_scores = []
        
        for user_id, rec_list in recommendations.items():
            if len(rec_list) < 2:
                continue
            
            # Calculate pairwise dissimilarity
            dissimilarities = []
            for i in range(len(rec_list)):
                for j in range(i + 1, len(rec_list)):
                    item_i = rec_list[i]
                    item_j = rec_list[j]
                    
                    if item_i in item_id_map and item_j in item_id_map:
                        idx_i = item_id_map[item_i]
                        idx_j = item_id_map[item_j]
                        similarity = item_similarity[idx_i, idx_j]
                        dissimilarity = 1 - similarity
                        dissimilarities.append(dissimilarity)
            
            if dissimilarities:
                diversity_scores.append(np.mean(dissimilarities))
        
        return np.mean(diversity_scores) if diversity_scores else 0.0
    
    def novelty(
        self,
        recommendations: Dict[int, List[int]],
        item_popularity: Dict[int, float]
    ) -> float:
        """
        Calculate average novelty (recommending less popular items).
        
        Args:
            recommendations: Dict mapping user_id to recommended items
            item_popularity: Dict mapping item_id to popularity score
            
        Returns:
            Average novelty score
        """
        novelty_scores = []
        
        for user_id, rec_list in recommendations.items():
            item_novelties = []
            for item in rec_list:
                if item in item_popularity:
                    # Novelty = -log(popularity)
                    popularity = item_popularity[item]
                    novelty = -np.log2(popularity + 1e-10)
                    item_novelties.append(novelty)
            
            if item_novelties:
                novelty_scores.append(np.mean(item_novelties))
        
        return np.mean(novelty_scores) if novelty_scores else 0.0
    
    def evaluate_model(
        self,
        recommender,
        test_data: pd.DataFrame,
        train_data: pd.DataFrame,
        all_items: Set[int],
        n_recommendations: int = 10,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Comprehensive evaluation of a recommender model.
        
        Args:
            recommender: Fitted recommender model
            test_data: Test ratings data
            train_data: Train ratings data (for excluding seen items)
            all_items: Set of all item IDs
            n_recommendations: Number of recommendations per user
            verbose: Whether to print results
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Prepare ground truth
        ground_truth = defaultdict(set)
        for _, row in test_data.iterrows():
            # Consider items with rating >= 4 as relevant
            if row['rating'] >= 4.0:
                ground_truth[row['user_id']].add(row['item_id'])
        
        # Get user's seen items from training data
        user_seen_items = defaultdict(set)
        for _, row in train_data.iterrows():
            user_seen_items[row['user_id']].add(row['item_id'])
        
        # Generate recommendations for all test users
        test_users = test_data['user_id'].unique()
        recommendations = {}
        
        if verbose:
            print("Generating recommendations...")
        
        for user_id in test_users:
            try:
                seen_items = user_seen_items[user_id]
                recs = recommender.recommend(
                    user_id, 
                    n=n_recommendations,
                    exclude_seen=True,
                    seen_items=seen_items
                )
                # Extract just item IDs
                recommendations[user_id] = [item_id for item_id, _ in recs]
            except:
                recommendations[user_id] = []
        
        # Calculate metrics
        results = {}
        
        # Ranking metrics at different K values
        for k in self.k_values:
            precision_scores = []
            recall_scores = []
            f1_scores = []
            ndcg_scores = []
            hit_scores = []
            
            for user_id in test_users:
                if user_id in ground_truth and user_id in recommendations:
                    recs = recommendations[user_id]
                    relevant = ground_truth[user_id]
                    
                    precision_scores.append(self.precision_at_k(recs, relevant, k))
                    recall_scores.append(self.recall_at_k(recs, relevant, k))
                    f1_scores.append(self.f1_at_k(recs, relevant, k))
                    ndcg_scores.append(self.ndcg_at_k(recs, relevant, k))
                    hit_scores.append(self.hit_rate_at_k(recs, relevant, k))
            
            results[f'Precision@{k}'] = np.mean(precision_scores)
            results[f'Recall@{k}'] = np.mean(recall_scores)
            results[f'F1@{k}'] = np.mean(f1_scores)
            results[f'NDCG@{k}'] = np.mean(ndcg_scores)
            results[f'HitRate@{k}'] = np.mean(hit_scores)
        
        # MAP
        results['MAP'] = self.mean_average_precision(recommendations, ground_truth)
        
        # Coverage
        results['Coverage'] = self.coverage(recommendations, all_items)
        
        # RMSE on test set
        rmse_scores = []
        for _, row in test_data.iterrows():
            try:
                pred = recommender.predict(row['user_id'], row['item_id'])
                error = (pred - row['rating']) ** 2
                rmse_scores.append(error)
            except:
                pass
        
        results['RMSE'] = np.sqrt(np.mean(rmse_scores)) if rmse_scores else float('inf')
        
        # Print results
        if verbose:
            print("\n" + "="*60)
            print("Evaluation Results")
            print("="*60)
            for metric, value in results.items():
                print(f"{metric:20s}: {value:.4f}")
        
        return results
    
    def compare_models(
        self,
        models: Dict[str, any],
        test_data: pd.DataFrame,
        train_data: pd.DataFrame,
        all_items: Set[int],
        n_recommendations: int = 10
    ) -> pd.DataFrame:
        """
        Compare multiple models.
        
        Args:
            models: Dict mapping model_name to fitted model
            test_data: Test data
            train_data: Train data
            all_items: Set of all items
            n_recommendations: Number of recommendations
            
        Returns:
            DataFrame with comparison results
        """
        results = {}
        
        for model_name, model in models.items():
            print(f"\nEvaluating {model_name}...")
            model_results = self.evaluate_model(
                model,
                test_data,
                train_data,
                all_items,
                n_recommendations,
                verbose=False
            )
            results[model_name] = model_results
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(results).T
        comparison_df = comparison_df.round(4)
        
        return comparison_df


if __name__ == "__main__":
    # Example usage
    from src.data_loader import MovieLensLoader
    from src.preprocess import prepare_data_for_training
    from src.recommenders.popularity import PopularityRecommender
    from src.recommenders.collaborative import ItemBasedCF
    
    # Load data
    loader = MovieLensLoader()
    ratings = loader.load_ratings()
    
    # Prepare data
    train, test, _ = prepare_data_for_training(
        ratings,
        test_size=0.2,
        split_method='user_based'
    )
    
    all_items = set(ratings['item_id'].unique())
    
    # Train models
    print("\nTraining models...")
    pop_model = PopularityRecommender(method='weighted')
    pop_model.fit(train)
    
    item_cf = ItemBasedCF(k=50)
    item_cf.fit(train)
    
    # Evaluate
    evaluator = RecommenderEvaluator(k_values=[5, 10, 20])
    
    results = evaluator.compare_models(
        models={
            'Popularity': pop_model,
            'ItemCF': item_cf
        },
        test_data=test,
        train_data=train,
        all_items=all_items,
        n_recommendations=10
    )
    
    print("\n" + "="*60)
    print("Model Comparison")
    print("="*60)
    print(results)