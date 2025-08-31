import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Tuple, Optional
import joblib
import os
from datetime import datetime

class GhibliFoodRecommendationEngine:
    def __init__(self, model_path: str = "models/"):
        self.model_path = model_path
        self.tfidf_vectorizer = None
        self.content_similarity_matrix = None
        self.svd_model = None
        self.scaler = StandardScaler()
        self.books_df = None
        self.user_ratings_df = None
        
    def prepare_content_features(self, books_data: List[Dict]) -> pd.DataFrame:
        """Prepare content-based features from books data"""
        df = pd.DataFrame(books_data)
        
        # Combine text features for content-based filtering
        df['combined_features'] = df.apply(lambda row: 
            f"{row.get('title', '')} {row.get('description', '')} "
            f"{row.get('genre', '')} {row.get('cuisineType', '')} "
            f"{row.get('dietaryCategory', '')} {row.get('difficultyLevel', '')}", 
            axis=1
        )
        
        # Handle ingredients if they exist
        if 'ingredients' in df.columns:
            df['ingredients_str'] = df['ingredients'].apply(
                lambda x: ' '.join(x) if isinstance(x, list) else str(x) if x else ''
            )
            df['combined_features'] += ' ' + df['ingredients_str']
        
        return df
    
    def train_content_based_model(self, books_data: List[Dict]):
        """Train content-based recommendation model"""
        print("Training content-based recommendation model...")
        
        self.books_df = self.prepare_content_features(books_data)
        
        # Create TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.8
        )
        
        # Fit and transform the combined features
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(
            self.books_df['combined_features'].fillna('')
        )
        
        # Calculate cosine similarity matrix
        self.content_similarity_matrix = cosine_similarity(tfidf_matrix)
        
        print(f"Content-based model trained with {len(books_data)} books")
    
    def train_collaborative_filtering_model(self, user_ratings_data: List[Dict]):
        """Train collaborative filtering model using matrix factorization"""
        print("Training collaborative filtering model...")
        
        # Convert user ratings to DataFrame
        ratings_df = pd.DataFrame(user_ratings_data)
        
        if ratings_df.empty:
            print("No rating data available for collaborative filtering")
            return
            
        # Create user-item matrix
        user_item_matrix = ratings_df.pivot_table(
            index='userId', 
            columns='bookId', 
            values='rating', 
            fill_value=0
        )
        
        self.user_ratings_df = user_item_matrix
        
        # Apply SVD for matrix factorization
        self.svd_model = TruncatedSVD(n_components=min(50, min(user_item_matrix.shape)-1))
        
        # Normalize the data
        user_item_normalized = self.scaler.fit_transform(user_item_matrix.fillna(0))
        
        # Fit SVD model
        self.svd_model.fit(user_item_normalized)
        
        print(f"Collaborative filtering model trained with {len(user_ratings_data)} ratings")
    
    def get_content_based_recommendations(
        self, 
        book_id: str, 
        n_recommendations: int = 5,
        exclude_book_ids: List[str] = None
    ) -> List[Dict]:
        """Get content-based recommendations for a given book"""
        if self.content_similarity_matrix is None or self.books_df is None:
            return []
        
        exclude_book_ids = exclude_book_ids or []
        
        try:
            # Find book index
            book_idx = self.books_df[self.books_df['id'] == book_id].index[0]
        except IndexError:
            print(f"Book with ID {book_id} not found")
            return []
        
        # Get similarity scores
        similarity_scores = list(enumerate(self.content_similarity_matrix[book_idx]))
        
        # Sort by similarity (excluding the book itself)
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[1:]
        
        recommendations = []
        for idx, score in similarity_scores:
            if len(recommendations) >= n_recommendations:
                break
                
            recommended_book = self.books_df.iloc[idx]
            
            # Skip if book should be excluded
            if recommended_book['id'] in exclude_book_ids:
                continue
                
            recommendations.append({
                'bookId': recommended_book['id'],
                'title': recommended_book.get('title', ''),
                'author': recommended_book.get('author', ''),
                'genre': recommended_book.get('genre', ''),
                'similarity_score': float(score),
                'recommendation_type': 'content_based',
                'reason': self._get_recommendation_reason(book_id, recommended_book['id'])
            })
        
        return recommendations
    
    def get_collaborative_recommendations(
        self, 
        user_id: str, 
        n_recommendations: int = 5,
        exclude_book_ids: List[str] = None
    ) -> List[Dict]:
        """Get collaborative filtering recommendations for a user"""
        if self.svd_model is None or self.user_ratings_df is None:
            return []
            
        exclude_book_ids = exclude_book_ids or []
        
        if user_id not in self.user_ratings_df.index:
            print(f"User {user_id} not found in ratings data")
            return []
        
        # Get user's rating vector
        user_ratings = self.user_ratings_df.loc[user_id].values.reshape(1, -1)
        
        # Transform using SVD
        user_ratings_normalized = self.scaler.transform(user_ratings)
        user_svd = self.svd_model.transform(user_ratings_normalized)
        
        # Get all items predictions
        all_items_svd = self.svd_model.components_
        predicted_ratings = np.dot(user_svd, all_items_svd)[0]
        
        # Get book IDs and create recommendations
        book_ids = self.user_ratings_df.columns.tolist()
        book_scores = list(zip(book_ids, predicted_ratings))
        
        # Filter out books user has already rated and excluded books
        user_rated_books = self.user_ratings_df.loc[user_id]
        unrated_books = [(book_id, score) for book_id, score in book_scores 
                        if user_rated_books[book_id] == 0 and book_id not in exclude_book_ids]
        
        # Sort by predicted rating
        unrated_books.sort(key=lambda x: x[1], reverse=True)
        
        recommendations = []
        for book_id, score in unrated_books[:n_recommendations]:
            # Get book details from books_df
            book_details = self.books_df[self.books_df['id'] == book_id]
            if not book_details.empty:
                book = book_details.iloc[0]
                recommendations.append({
                    'bookId': book_id,
                    'title': book.get('title', ''),
                    'author': book.get('author', ''),
                    'genre': book.get('genre', ''),
                    'predicted_rating': float(score),
                    'recommendation_type': 'collaborative_filtering'
                })
        
        return recommendations
    
    def get_hybrid_recommendations(
        self, 
        user_id: str, 
        liked_book_ids: List[str] = None,
        n_recommendations: int = 5,
        exclude_book_ids: List[str] = None,
        content_weight: float = 0.6,
        collaborative_weight: float = 0.4
    ) -> List[Dict]:
        """Get hybrid recommendations combining content-based and collaborative filtering"""
        exclude_book_ids = exclude_book_ids or []
        liked_book_ids = liked_book_ids or []
        
        all_recommendations = {}
        
        # Get collaborative filtering recommendations
        collaborative_recs = self.get_collaborative_recommendations(
            user_id, n_recommendations * 2, exclude_book_ids
        )
        
        # Add collaborative recommendations to the mix
        for rec in collaborative_recs:
            book_id = rec['bookId']
            score = rec.get('predicted_rating', 0) * collaborative_weight
            all_recommendations[book_id] = {
                **rec,
                'hybrid_score': score,
                'collaborative_score': rec.get('predicted_rating', 0)
            }
        
        # Get content-based recommendations from liked books
        for liked_book_id in liked_book_ids:
            content_recs = self.get_content_based_recommendations(
                liked_book_id, n_recommendations, exclude_book_ids
            )
            
            for rec in content_recs:
                book_id = rec['bookId']
                content_score = rec.get('similarity_score', 0) * content_weight
                
                if book_id in all_recommendations:
                    # Combine scores
                    all_recommendations[book_id]['hybrid_score'] += content_score
                    all_recommendations[book_id]['content_score'] = rec.get('similarity_score', 0)
                else:
                    all_recommendations[book_id] = {
                        **rec,
                        'hybrid_score': content_score,
                        'content_score': rec.get('similarity_score', 0)
                    }
        
        # Sort by hybrid score and return top recommendations
        sorted_recommendations = sorted(
            all_recommendations.values(),
            key=lambda x: x['hybrid_score'],
            reverse=True
        )
        
        return sorted_recommendations[:n_recommendations]
    
    def _get_recommendation_reason(self, source_book_id: str, recommended_book_id: str) -> str:
        """Generate explanation for why a book is recommended"""
        if self.books_df is None:
            return "Similar content features"
            
        try:
            source_book = self.books_df[self.books_df['id'] == source_book_id].iloc[0]
            recommended_book = self.books_df[self.books_df['id'] == recommended_book_id].iloc[0]
            
            reasons = []
            
            if source_book.get('genre') == recommended_book.get('genre'):
                reasons.append(f"same genre ({source_book.get('genre')})")
            
            if source_book.get('cuisineType') == recommended_book.get('cuisineType'):
                reasons.append(f"same cuisine type ({source_book.get('cuisineType')})")
            
            if source_book.get('difficultyLevel') == recommended_book.get('difficultyLevel'):
                reasons.append(f"same difficulty level ({source_book.get('difficultyLevel')})")
            
            if reasons:
                return f"Recommended because of {', '.join(reasons)}"
            else:
                return "Similar content and themes"
                
        except Exception:
            return "Similar content features"
    
    def save_models(self):
        """Save trained models to disk"""
        os.makedirs(self.model_path, exist_ok=True)
        
        if self.tfidf_vectorizer:
            joblib.dump(self.tfidf_vectorizer, f"{self.model_path}/tfidf_vectorizer.pkl")
        
        if self.content_similarity_matrix is not None:
            np.save(f"{self.model_path}/content_similarity_matrix.npy", self.content_similarity_matrix)
        
        if self.svd_model:
            joblib.dump(self.svd_model, f"{self.model_path}/svd_model.pkl")
            joblib.dump(self.scaler, f"{self.model_path}/scaler.pkl")
        
        if self.books_df is not None:
            self.books_df.to_pickle(f"{self.model_path}/books_df.pkl")
        
        if self.user_ratings_df is not None:
            self.user_ratings_df.to_pickle(f"{self.model_path}/user_ratings_df.pkl")
        
        # Save metadata
        metadata = {
            'last_trained': datetime.now().isoformat(),
            'model_version': '1.0.0',
            'books_count': len(self.books_df) if self.books_df is not None else 0,
            'ratings_count': len(self.user_ratings_df) if self.user_ratings_df is not None else 0
        }
        joblib.dump(metadata, f"{self.model_path}/metadata.pkl")
        
        print(f"Models saved to {self.model_path}")
    
    def load_models(self):
        """Load trained models from disk"""
        try:
            if os.path.exists(f"{self.model_path}/tfidf_vectorizer.pkl"):
                self.tfidf_vectorizer = joblib.load(f"{self.model_path}/tfidf_vectorizer.pkl")
            
            if os.path.exists(f"{self.model_path}/content_similarity_matrix.npy"):
                self.content_similarity_matrix = np.load(f"{self.model_path}/content_similarity_matrix.npy")
            
            if os.path.exists(f"{self.model_path}/svd_model.pkl"):
                self.svd_model = joblib.load(f"{self.model_path}/svd_model.pkl")
                self.scaler = joblib.load(f"{self.model_path}/scaler.pkl")
            
            if os.path.exists(f"{self.model_path}/books_df.pkl"):
                self.books_df = pd.read_pickle(f"{self.model_path}/books_df.pkl")
            
            if os.path.exists(f"{self.model_path}/user_ratings_df.pkl"):
                self.user_ratings_df = pd.read_pickle(f"{self.model_path}/user_ratings_df.pkl")
            
            print("Models loaded successfully")
            return True
        except Exception as e:
            print(f"Error loading models: {e}")
            return False