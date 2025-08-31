from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from models.recommendation_engine import GhibliFoodRecommendationEngine
from utils.data_fetcher import DataFetcher
from config.settings import settings
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

# Global recommendation engine instance
recommendation_engine = GhibliFoodRecommendationEngine(settings.MODEL_PATH)

# Pydantic models for request/response
class RecommendationRequest(BaseModel):
    user_id: str
    liked_book_ids: Optional[List[str]] = []
    exclude_book_ids: Optional[List[str]] = []
    n_recommendations: Optional[int] = Field(default=5, ge=1, le=20)
    recommendation_type: Optional[str] = Field(default="hybrid", regex="^(content|collaborative|hybrid)$")

class BookRecommendationRequest(BaseModel):
    book_id: str
    exclude_book_ids: Optional[List[str]] = []
    n_recommendations: Optional[int] = Field(default=5, ge=1, le=20)

class RecommendationResponse(BaseModel):
    bookId: str
    title: str
    author: str
    genre: str
    similarity_score: Optional[float] = None
    predicted_rating: Optional[float] = None
    hybrid_score: Optional[float] = None
    recommendation_type: str
    reason: Optional[str] = None

class TrainingRequest(BaseModel):
    use_mock_data: Optional[bool] = True
    force_retrain: Optional[bool] = False

# Dependency to get recommendation engine
def get_recommendation_engine():
    return recommendation_engine

@router.get("/recommendations/health")
async def health_check():
    """Check if recommendation service is healthy"""
    try:
        models_loaded = recommendation_engine.load_models()
        return JSONResponse(content={
            "status": "healthy",
            "models_loaded": models_loaded,
            "service": "recommendation-engine"
        })
    except Exception as e:
        return JSONResponse(
            content={"status": "unhealthy", "error": str(e)},
            status_code=500
        )

@router.post("/recommendations/train")
async def train_models(
    request: TrainingRequest,
    background_tasks: BackgroundTasks,
    engine: GhibliFoodRecommendationEngine = Depends(get_recommendation_engine)
):
    """Train or retrain the recommendation models"""
    try:
        # Check if models already exist and force_retrain is False
        if not request.force_retrain:
            if engine.load_models():
                return JSONResponse(content={
                    "message": "Models already trained and loaded",
                    "status": "loaded_existing"
                })
        
        # Start training in background
        background_tasks.add_task(
            _train_models_background,
            engine,
            request.use_mock_data
        )
        
        return JSONResponse(content={
            "message": "Model training started",
            "status": "training_initiated"
        })
        
    except Exception as e:
        logger.error(f"Error starting model training: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def _train_models_background(engine: GhibliFoodRecommendationEngine, use_mock_data: bool):
    """Background task to train models"""
    try:
        data_fetcher = DataFetcher()
        books_data, ratings_data = await data_fetcher.get_training_data(use_mock_data)
        
        # Train content-based model
        engine.train_content_based_model(books_data)
        
        # Train collaborative filtering model if we have ratings
        if ratings_data:
            engine.train_collaborative_filtering_model(ratings_data)
        
        # Save models
        engine.save_models()
        
        logger.info("Model training completed successfully")
        
    except Exception as e:
        logger.error(f"Error in background model training: {e}")

@router.post("/recommendations/user", response_model=List[RecommendationResponse])
async def get_user_recommendations(
    request: RecommendationRequest,
    engine: GhibliFoodRecommendationEngine = Depends(get_recommendation_engine)
):
    """Get personalized recommendations for a user"""
    try:
        # Ensure models are loaded
        if not engine.load_models():
            raise HTTPException(
                status_code=503, 
                detail="Models not trained yet. Please train models first."
            )
        
        recommendations = []
        
        if request.recommendation_type == "content" or request.recommendation_type == "hybrid":
            # Get content-based recommendations from liked books
            for book_id in request.liked_book_ids:
                content_recs = engine.get_content_based_recommendations(
                    book_id=book_id,
                    n_recommendations=request.n_recommendations,
                    exclude_book_ids=request.exclude_book_ids
                )
                recommendations.extend(content_recs)
        
        if request.recommendation_type == "collaborative":
            # Get collaborative filtering recommendations
            collaborative_recs = engine.get_collaborative_recommendations(
                user_id=request.user_id,
                n_recommendations=request.n_recommendations,
                exclude_book_ids=request.exclude_book_ids
            )
            recommendations.extend(collaborative_recs)
        
        if request.recommendation_type == "hybrid":
            # Get hybrid recommendations
            hybrid_recs = engine.get_hybrid_recommendations(
                user_id=request.user_id,
                liked_book_ids=request.liked_book_ids,
                n_recommendations=request.n_recommendations,
                exclude_book_ids=request.exclude_book_ids
            )
            recommendations = hybrid_recs
        
        # Remove duplicates and limit results
        seen_books = set()
        unique_recommendations = []
        for rec in recommendations:
            if rec['bookId'] not in seen_books:
                seen_books.add(rec['bookId'])
                unique_recommendations.append(rec)
                if len(unique_recommendations) >= request.n_recommendations:
                    break
        
        return unique_recommendations
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting user recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/recommendations/book", response_model=List[RecommendationResponse])
async def get_book_recommendations(
    request: BookRecommendationRequest,
    engine: GhibliFoodRecommendationEngine = Depends(get_recommendation_engine)
):
    """Get recommendations based on a specific book (similar books)"""
    try:
        # Ensure models are loaded
        if not engine.load_models():
            raise HTTPException(
                status_code=503,
                detail="Models not trained yet. Please train models first."
            )
        
        recommendations = engine.get_content_based_recommendations(
            book_id=request.book_id,
            n_recommendations=request.n_recommendations,
            exclude_book_ids=request.exclude_book_ids
        )
        
        return recommendations
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting book recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/recommendations/popular")
async def get_popular_books(
    n_books: int = 10,
    genre: Optional[str] = None
):
    """Get popular books based on ratings and reviews"""
    try:
        data_fetcher = DataFetcher()
        books = await data_fetcher.fetch_all_books()
        
        # Filter by genre if specified
        if genre:
            books = [book for book in books if book.get('genre', '').lower() == genre.lower()]
        
        # Sort by a popularity score (you can customize this logic)
        # For now, using a simple scoring based on ratings and reviews
        for book in books:
            # Mock popularity score calculation
            avg_rating = book.get('averageRating', 3.0)
            review_count = book.get('reviewCount', 0)
            
            # Simple popularity score: weighted average with review count
            popularity_score = (avg_rating * 0.7) + (min(review_count / 10, 5) * 0.3)
            book['popularity_score'] = popularity_score
        
        # Sort by popularity score
        popular_books = sorted(books, key=lambda x: x.get('popularity_score', 0), reverse=True)
        
        return {
            "popular_books": popular_books[:n_books],
            "total_books": len(books)
        }
        
    except Exception as e:
        logger.error(f"Error getting popular books: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/recommendations/trending")
async def get_trending_books(n_books: int = 10):
    """Get trending books (recently added or highly rated)"""
    try:
        data_fetcher = DataFetcher()
        books = await data_fetcher.fetch_all_books()
        
        # For trending, we'll prioritize recently added books with good ratings
        # In a real system, you'd factor in recent user activity
        
        # Sort by creation date (assuming newer books are more trending)
        # This is a simplified approach
        trending_books = books[:n_books]  # Take the first n books as "trending"
        
        return {
            "trending_books": trending_books,
            "total_books": len(books)
        }
        
    except Exception as e:
        logger.error(f"Error getting trending books: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/recommendations/stats")
async def get_recommendation_stats():
    """Get statistics about the recommendation system"""
    try:
        data_fetcher = DataFetcher()
        books = await data_fetcher.fetch_all_books()
        
        # Calculate basic statistics
        total_books = len(books)
        genres = {}
        cuisine_types = {}
        difficulty_levels = {}
        
        for book in books:
            # Count genres
            genre = book.get('genre', 'Unknown')
            genres[genre] = genres.get(genre, 0) + 1
            
            # Count cuisine types
            cuisine_type = book.get('cuisineType', 'Unknown')
            cuisine_types[cuisine_type] = cuisine_types.get(cuisine_type, 0) + 1
            
            # Count difficulty levels
            difficulty = book.get('difficultyLevel', 'Unknown')
            difficulty_levels[difficulty] = difficulty_levels.get(difficulty, 0) + 1
        
        return {
            "total_books": total_books,
            "genre_distribution": genres,
            "cuisine_type_distribution": cuisine_types,
            "difficulty_distribution": difficulty_levels,
            "models_trained": recommendation_engine.content_similarity_matrix is not None,
        }
        
    except Exception as e:
        logger.error(f"Error getting recommendation stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))