# Machine-Learning-Web - Ghibli Food Recipe Platform

A comprehensive machine learning recommendation system providing intelligent book suggestions, enhanced search capabilities, and user behavior analysis for the Ghibli Food Recipe application ecosystem.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Integration](#project-integration)
- [Quick Setup Guide](#quick-setup-guide)
- [Technology Stack](#technology-stack)
- [API Documentation](#api-documentation)
- [Configuration](#configuration)
- [Integration Examples](#integration-examples)
- [Development](#development)
- [Contributing](#contributing)

---

## Overview

The **Machine-Learning-Web** component serves as the intelligent recommendation engine for the entire Ghibli Food Recipe platform, providing AI-powered book recommendations, enhanced search capabilities, and user behavior analysis. This service uses advanced machine learning algorithms to deliver personalized experiences and improve user engagement through data-driven insights.

This solution includes content-based and collaborative filtering algorithms, hybrid recommendation systems, real-time model training, behavioral pattern recognition, and scalable FastAPI architecture optimized for high-throughput recommendation serving.

---

## Features

### Core Features
- **Hybrid Recommendation System** - Combines content-based and collaborative filtering for optimal accuracy
- **Real-time API Service** - FastAPI-based service with automatic OpenAPI documentation
- **Smart Search Enhancement** - Semantic similarity and relevance scoring for improved search results
- **User Behavior Analysis** - Pattern recognition and preference learning from user interactions
- **Scalable Architecture** - Designed for high-throughput recommendation serving with caching

### Advanced Features
- **Cold Start Problem Solution** - Handles new users and books with no interaction history
- **A/B Testing Framework** - Support for testing different recommendation algorithms
- **Model Monitoring** - Performance tracking, accuracy metrics, and model drift detection
- **Feature Engineering** - Automated extraction and processing of book and user features
- **Continuous Learning** - Real-time model updates based on user feedback and interactions

---

## Project Integration

This ML service acts as the intelligence layer for the entire Ghibli Food Recipe platform:

### 🔧 **Backend Integration** (Back-End-Web)
- **Recommendation API** - Provides personalized book recommendations via REST endpoints
- **Behavioral Data Pipeline** - Receives and processes user interaction data for model training
- **Search Enhancement** - Integrates with backend search to provide ML-powered relevance scoring
- **Real-time Processing** - Handles new books and users immediately for instant recommendations

### 🎨 **Frontend Integration** (Front-End-Web)
- **Recommendation Widgets** - Powers recommendation sections with personalized content
- **Smart Search Interface** - Enhances search results with ML-driven relevance and suggestions
- **User Experience Optimization** - Adapts interface elements based on user preference patterns
- **Behavioral Tracking Integration** - Seamlessly captures user interactions for model improvement

### 🗄️ **Database Integration** (Database-Web)
- **Feature Storage** - Caches computed features and embeddings for fast recommendation serving
- **Model Persistence** - Stores trained models, user profiles, and recommendation histories
- **Analytics Integration** - Accesses user interaction history and book metadata for training
- **Performance Metrics** - Logs recommendation accuracy, user feedback, and system performance

### 🚀 **DevOps Integration** (DevOps-Web)
- **Container Deployment** - Docker-ready with health checks and resource monitoring
- **MLOps Pipeline** - Automated model training, validation, and deployment workflows
- **Monitoring Integration** - Prometheus metrics collection and Grafana visualization
- **Auto-scaling Support** - Kubernetes HPA integration for handling variable ML workloads

---

## Quick Setup Guide

### Prerequisites
- **Python 3.11+** with pip and virtual environment support
- **PostgreSQL** database (shared with other services)
- **Redis** (optional, for caching and performance optimization)
- **Docker** (optional, for containerized deployment)

### Local Development Setup

1. **Environment Setup**
   ```bash
   cd Machine-Learnimg-Web/Ghibli-Food-ML
   python -m venv venv
   
   # Activate virtual environment
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install dependencies
   pip install -r requirements.txt
   ```

2. **Environment Configuration**
   ```bash
   cp .env.example .env
   ```
   
   Configure the following variables:
   ```env
   # API Configuration
   API_HOST=0.0.0.0
   API_PORT=8001
   
   # Database Connection
   DATABASE_URL=postgresql://ghibli_api_user:password@localhost:5432/ghibli_food_db
   
   # Backend API Integration
   BACKEND_API_URL=http://localhost:5000/api/v1
   
   # Redis Cache
   REDIS_URL=redis://localhost:6379/0
   
   # ML Model Configuration
   MODEL_UPDATE_INTERVAL=3600
   RECOMMENDATION_CACHE_TTL=1800
   FEATURE_VECTOR_SIZE=100
   
   # Training Parameters
   MIN_INTERACTIONS_PER_USER=5
   MIN_INTERACTIONS_PER_BOOK=3
   COLLABORATIVE_FACTORS=50
   CONTENT_WEIGHT=0.4
   COLLABORATIVE_WEIGHT=0.6
   ```

3. **Start the ML Service**
   ```bash
   python src/main.py
   ```
   
   Access the API documentation at: `http://localhost:8001/docs`

4. **Verify Installation**
   ```bash
   # Health check
   curl http://localhost:8001/health
   
   # Model status
   curl http://localhost:8001/models/status
   ```

### Integrated Development Setup

1. **Start All Services**
   ```bash
   # Terminal 1: Database
   cd Database-Web/Ghibli-Food-Database
   npm run dev
   
   # Terminal 2: Backend API  
   cd Back-End-Web/Ghibli-Food-Receipt-API
   npm run dev
   
   # Terminal 3: ML Service
   cd Machine-Learnimg-Web/Ghibli-Food-ML
   python src/main.py
   
   # Terminal 4: Frontend
   cd Front-End-Web/Ghibli-Food-Receipt
   npm run dev
   ```

### Docker Integration
```bash
# Using integrated Docker setup
cd DevOps-Web/Ghibli-Food-DevOps
docker-compose up -d ml-service
```

---

## Technology Stack

| Technology | Version | Purpose |
|------------|---------|---------|
| **Python** | 3.11+ | Core programming language |
| **FastAPI** | 0.104+ | High-performance web framework |
| **scikit-learn** | 1.3+ | Machine learning algorithms |
| **pandas** | 2.0+ | Data manipulation and analysis |
| **numpy** | 1.24+ | Numerical computing |
| **SQLAlchemy** | 2.0+ | Database ORM |

### Machine Learning Libraries
| Technology | Purpose |
|------------|---------|
| **scikit-learn** | ML algorithms (SVD, TF-IDF, clustering) |
| **surprise** | Collaborative filtering algorithms |
| **sentence-transformers** | Semantic text embeddings |
| **joblib** | Model serialization and parallel processing |

### Data & Infrastructure
| Technology | Purpose |
|------------|---------|
| **PostgreSQL** | Primary data storage |
| **Redis** | Caching and session storage |
| **uvicorn** | ASGI server for FastAPI |
| **pydantic** | Data validation and serialization |

### Development & Testing
| Technology | Purpose |
|------------|---------|
| **pytest** | Testing framework |
| **black** | Code formatting |
| **flake8** | Code linting |
| **mypy** | Static type checking |

---

## API Documentation

### Recommendation Endpoints

**POST** `/recommend` - Get personalized recommendations
```json
Request:
{
  "user_id": "uuid",
  "num_recommendations": 10,
  "preferences": {
    "genres": ["cookbook", "asian"],
    "difficulty": "intermediate", 
    "exclude_read": true
  }
}

Response:
{
  "recommendations": [
    {
      "book_id": "uuid",
      "title": "Book Title",
      "score": 0.85,
      "reason": "Based on your interest in Asian cuisine"
    }
  ],
  "algorithm_used": "hybrid",
  "cache_hit": false
}
```

**GET** `/similar/{book_id}` - Get similar books
```json
Response:
{
  "similar_books": [
    {
      "book_id": "uuid",
      "title": "Similar Book",
      "similarity_score": 0.92,
      "common_features": ["genre", "author_style"]
    }
  ]
}
```

### Search Endpoints

**POST** `/search` - Enhanced semantic search
```json
Request:
{
  "query": "quick pasta recipes",
  "filters": {
    "genres": ["italian"],
    "max_difficulty": "intermediate"
  },
  "limit": 20
}

Response:
{
  "results": [
    {
      "book_id": "uuid",
      "title": "Italian Pasta Mastery",
      "relevance_score": 0.95,
      "matched_terms": ["pasta", "quick", "recipes"]
    }
  ],
  "total_results": 45,
  "search_time_ms": 23
}
```

**GET** `/search/suggestions` - Auto-complete suggestions
```json
Response:
{
  "suggestions": [
    "chicken curry recipes",
    "chicken soup cookbook", 
    "chicken cooking techniques"
  ]
}
```

### Analytics Endpoints

**POST** `/behavior` - Track user behavior
```json
Request:
{
  "user_id": "uuid",
  "book_id": "uuid",
  "action": "view|rate|add_to_library|purchase",
  "metadata": {
    "rating": 5,
    "time_spent": 120,
    "page_views": 3
  },
  "timestamp": "2024-01-01T00:00:00Z"
}
```

**GET** `/analytics/user/{user_id}` - User analytics and insights
```json
Response:
{
  "user_profile": {
    "preferred_genres": ["asian", "vegetarian"],
    "avg_rating_given": 4.2,
    "behavior_patterns": {
      "most_active_time": "evening",
      "preferred_difficulty": "intermediate"
    },
    "recommendation_accuracy": 0.85
  }
}
```

### Model Management Endpoints

**POST** `/models/train` - Trigger model retraining
**GET** `/models/status` - Model performance and health metrics
**GET** `/health` - Service health check and system status

---

## Configuration

### Environment Variables

```env
# API Server Configuration
API_HOST=0.0.0.0
API_PORT=8001
API_WORKERS=4

# Database Configuration
DATABASE_URL=postgresql://ghibli_api_user:password@localhost:5432/ghibli_food_db
DB_POOL_SIZE=10
DB_POOL_OVERFLOW=20

# Cache Configuration  
REDIS_URL=redis://localhost:6379/0
CACHE_TTL_RECOMMENDATIONS=1800
CACHE_TTL_FEATURES=3600

# ML Model Parameters
MODEL_UPDATE_INTERVAL=3600
RECOMMENDATION_BATCH_SIZE=1000
FEATURE_VECTOR_SIZE=100
MIN_INTERACTIONS_PER_USER=5
MIN_INTERACTIONS_PER_BOOK=3

# Collaborative Filtering
COLLABORATIVE_FACTORS=50
COLLABORATIVE_ITERATIONS=20
COLLABORATIVE_REGULARIZATION=0.02

# Content-Based Filtering
TFIDF_MAX_FEATURES=10000
TFIDF_NGRAM_RANGE_MIN=1
TFIDF_NGRAM_RANGE_MAX=2

# Hybrid Model Weights
CONTENT_WEIGHT=0.4
COLLABORATIVE_WEIGHT=0.6
POPULARITY_WEIGHT=0.1

# Performance Tuning
SIMILARITY_CACHE_SIZE=10000
MODEL_CACHE_SIZE=5
ENABLE_PARALLEL_PROCESSING=true
MAX_WORKERS=4

# Monitoring and Logging
LOG_LEVEL=INFO
ENABLE_METRICS=true
METRICS_PORT=8002
```

### Model Configuration
```python
# config/models.py
MODEL_CONFIG = {
    "collaborative_filtering": {
        "algorithm": "SVD",
        "n_factors": 50,
        "n_epochs": 20,
        "lr_all": 0.005,
        "reg_all": 0.02
    },
    "content_based": {
        "tfidf_max_features": 10000,
        "ngram_range": (1, 2),
        "stop_words": "english",
        "min_df": 2,
        "max_df": 0.95
    },
    "hybrid": {
        "content_weight": 0.4,
        "collaborative_weight": 0.6,
        "popularity_weight": 0.1
    }
}
```

---

## Integration Examples

### Recommendation Service
```python
# services/recommendation_service.py
from typing import List, Dict, Any
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class RecommendationService:
    def __init__(self):
        self.content_recommender = ContentBasedRecommender()
        self.collaborative_recommender = CollaborativeRecommender()
        self.hybrid_recommender = HybridRecommender()
    
    async def get_recommendations(
        self, 
        user_id: str, 
        num_recommendations: int = 10,
        preferences: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """Get personalized recommendations for a user."""
        
        # Check cache first
        cache_key = f"recommendations:{user_id}:{hash(str(preferences))}"
        cached_recommendations = await self.cache.get(cache_key)
        
        if cached_recommendations:
            return cached_recommendations
        
        # Get user interaction history
        user_interactions = await self.get_user_interactions(user_id)
        
        # Choose recommendation strategy based on data availability
        if len(user_interactions) < 5:
            # Cold start: use content-based + popularity
            recommendations = await self.handle_cold_start(
                user_id, preferences, num_recommendations
            )
        else:
            # Use hybrid approach
            recommendations = await self.hybrid_recommender.recommend(
                user_id, num_recommendations, preferences
            )
        
        # Cache results
        await self.cache.set(cache_key, recommendations, ttl=1800)
        
        return recommendations

class ContentBasedRecommender:
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.feature_matrix = None
        self.similarity_matrix = None
    
    async def train(self, books_df):
        """Train content-based model on book features."""
        # Combine textual features
        content_features = (
            books_df['title'].fillna('') + ' ' +
            books_df['description'].fillna('') + ' ' +
            books_df['genre'].fillna('') + ' ' +
            books_df['ingredients'].apply(lambda x: ' '.join(x) if x else '')
        )
        
        # Create TF-IDF features
        self.feature_matrix = self.tfidf_vectorizer.fit_transform(content_features)
        self.similarity_matrix = cosine_similarity(self.feature_matrix)
        
        logger.info(f"Content-based model trained with {len(books_df)} books")
    
    async def get_similar_books(self, book_id: str, limit: int = 10):
        """Get books similar to the given book."""
        book_idx = self.book_id_to_index.get(book_id)
        if book_idx is None:
            return []
        
        similarity_scores = self.similarity_matrix[book_idx]
        similar_indices = similarity_scores.argsort()[-limit-1:-1][::-1]
        
        recommendations = []
        for idx in similar_indices:
            if idx != book_idx:
                recommendations.append({
                    'book_id': self.index_to_book_id[idx],
                    'similarity_score': float(similarity_scores[idx]),
                    'algorithm': 'content_based'
                })
        
        return recommendations
```

### User Behavior Tracking
```python
# services/behavior_service.py
from datetime import datetime
from typing import Dict, Any

class BehaviorTrackingService:
    def __init__(self, db_session, ml_service):
        self.db = db_session
        self.ml_service = ml_service
    
    async def track_behavior(
        self, 
        user_id: str, 
        book_id: str, 
        action: str, 
        metadata: Dict[str, Any] = None
    ):
        """Track user behavior and update ML models."""
        
        # Store interaction in database
        interaction = UserInteraction(
            user_id=user_id,
            book_id=book_id,
            action=action,
            metadata=metadata or {},
            timestamp=datetime.utcnow()
        )
        
        self.db.add(interaction)
        await self.db.commit()
        
        # Update user profile
        await self.update_user_profile(user_id, action, metadata)
        
        # Trigger incremental model update
        if action in ['rate', 'add_to_library', 'purchase']:
            await self.ml_service.update_user_factors(user_id, book_id, action, metadata)
        
        # Invalidate recommendation cache
        await self.invalidate_user_cache(user_id)
    
    async def update_user_profile(self, user_id: str, action: str, metadata: Dict[str, Any]):
        """Update user profile based on behavior."""
        user_profile = await self.get_or_create_user_profile(user_id)
        
        # Update interaction counts
        user_profile.total_interactions += 1
        user_profile.action_counts[action] = user_profile.action_counts.get(action, 0) + 1
        
        # Update preferences based on action
        if action == 'rate' and metadata.get('rating'):
            rating = metadata['rating']
            user_profile.average_rating = (
                (user_profile.average_rating * user_profile.rating_count + rating) /
                (user_profile.rating_count + 1)
            )
            user_profile.rating_count += 1
        
        # Update time-based patterns
        current_hour = datetime.utcnow().hour
        user_profile.active_hours[current_hour] = user_profile.active_hours.get(current_hour, 0) + 1
        
        await self.db.commit()

class RealtimeModelUpdater:
    def __init__(self):
        self.update_queue = asyncio.Queue()
        self.batch_size = 100
        self.update_interval = 300  # 5 minutes
    
    async def start_update_worker(self):
        """Background worker for model updates."""
        while True:
            try:
                # Collect batch of updates
                updates = []
                for _ in range(self.batch_size):
                    try:
                        update = await asyncio.wait_for(
                            self.update_queue.get(), timeout=self.update_interval
                        )
                        updates.append(update)
                    except asyncio.TimeoutError:
                        break
                
                if updates:
                    await self.process_batch_update(updates)
                    
            except Exception as e:
                logger.error(f"Model update worker error: {e}")
                await asyncio.sleep(60)  # Wait before retry
    
    async def process_batch_update(self, updates: List[Dict[str, Any]]):
        """Process a batch of model updates."""
        logger.info(f"Processing batch update with {len(updates)} interactions")
        
        # Group updates by user
        user_updates = {}
        for update in updates:
            user_id = update['user_id']
            if user_id not in user_updates:
                user_updates[user_id] = []
            user_updates[user_id].append(update)
        
        # Update collaborative filtering model
        for user_id, user_interactions in user_updates.items():
            await self.update_user_embeddings(user_id, user_interactions)
        
        # Update content-based features if new books
        new_books = [u for u in updates if u.get('new_book')]
        if new_books:
            await self.update_content_features(new_books)
```

### Performance Monitoring
```python
# monitoring/metrics.py
from prometheus_client import Counter, Histogram, Gauge
import time
from functools import wraps

# Metrics
RECOMMENDATION_REQUESTS = Counter('ml_recommendation_requests_total', 'Total recommendation requests')
RECOMMENDATION_LATENCY = Histogram('ml_recommendation_duration_seconds', 'Recommendation request duration')
MODEL_ACCURACY = Gauge('ml_model_accuracy', 'Model accuracy score')
CACHE_HIT_RATE = Counter('ml_cache_hits_total', 'Cache hits')

def monitor_performance(func):
    """Decorator to monitor function performance."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            RECOMMENDATION_REQUESTS.inc()
            return result
        finally:
            RECOMMENDATION_LATENCY.observe(time.time() - start_time)
    return wrapper

class ModelPerformanceMonitor:
    def __init__(self):
        self.accuracy_history = []
        self.precision_history = []
        self.recall_history = []
    
    async def evaluate_model_performance(self):
        """Evaluate and log model performance metrics."""
        # Get test interactions
        test_interactions = await self.get_test_interactions()
        
        # Calculate metrics
        accuracy = await self.calculate_accuracy(test_interactions)
        precision = await self.calculate_precision_at_k(test_interactions, k=10)
        recall = await self.calculate_recall_at_k(test_interactions, k=10)
        
        # Update metrics
        MODEL_ACCURACY.set(accuracy)
        
        # Log to history
        self.accuracy_history.append(accuracy)
        self.precision_history.append(precision)
        self.recall_history.append(recall)
        
        # Alert if performance drops
        if len(self.accuracy_history) > 5:
            recent_avg = sum(self.accuracy_history[-5:]) / 5
            if recent_avg < 0.7:  # Threshold
                await self.send_performance_alert(recent_avg)
        
        logger.info(f"Model performance - Accuracy: {accuracy:.3f}, Precision@10: {precision:.3f}, Recall@10: {recall:.3f}")
```

---

## Development

### Development Scripts
```bash
# Environment setup
python -m venv venv             # Create virtual environment
source venv/bin/activate        # Activate (Linux/Mac)
pip install -r requirements.txt # Install dependencies

# Development server
python src/main.py              # Start FastAPI server
uvicorn main:app --reload       # Auto-reload on changes
uvicorn main:app --workers 4    # Multi-worker setup

# Model training and evaluation
python scripts/train_models.py  # Train all models
python scripts/evaluate_models.py # Evaluate performance
python scripts/update_features.py # Update feature cache

# Testing
pytest                          # Run all tests
pytest -v tests/test_api.py     # Verbose API tests
pytest --cov=src tests/         # Coverage report
pytest -k "recommendation"      # Run specific tests

# Code quality
black src/                      # Format code
flake8 src/                     # Lint code
mypy src/                       # Type checking
isort src/                      # Sort imports
```

### Testing Strategy
- **Unit Tests**: Individual ML algorithm and utility function testing
- **Integration Tests**: API endpoint testing with database integration
- **Model Tests**: Algorithm accuracy and performance validation
- **Load Tests**: High-throughput recommendation serving testing
- **A/B Tests**: Comparative algorithm performance testing

### Model Development Workflow
```python
# Example model training pipeline
async def train_recommendation_models():
    """Complete model training pipeline."""
    
    # 1. Data extraction
    interactions_df = await extract_user_interactions()
    books_df = await extract_book_features()
    
    # 2. Data preprocessing
    interactions_df = preprocess_interactions(interactions_df)
    books_df = preprocess_book_features(books_df)
    
    # 3. Train models
    content_model = await train_content_based_model(books_df)
    collab_model = await train_collaborative_model(interactions_df)
    
    # 4. Evaluate models
    content_metrics = await evaluate_content_model(content_model)
    collab_metrics = await evaluate_collaborative_model(collab_model)
    
    # 5. Create hybrid model
    hybrid_model = create_hybrid_model(content_model, collab_model)
    hybrid_metrics = await evaluate_hybrid_model(hybrid_model)
    
    # 6. Model deployment
    if hybrid_metrics['accuracy'] > 0.75:
        await deploy_model(hybrid_model)
        logger.info(f"Model deployed with accuracy: {hybrid_metrics['accuracy']}")
    else:
        logger.warning(f"Model accuracy too low: {hybrid_metrics['accuracy']}")
```

---

## Contributing

1. **Code Standards**
   - Follow PEP 8 style guidelines
   - Use type hints for all function parameters and returns
   - Implement comprehensive docstrings with examples
   - Maintain 90%+ test coverage for ML algorithms

2. **ML Model Guidelines**
   - Document algorithm choices and hyperparameters
   - Include model performance benchmarks
   - Implement proper cross-validation techniques
   - Use reproducible random seeds

3. **API Development**
   - Follow FastAPI best practices
   - Implement proper request validation with Pydantic
   - Include comprehensive API documentation
   - Handle errors gracefully with meaningful responses

4. **Performance Requirements**
   - Recommendation latency < 100ms for cached results
   - Recommendation latency < 500ms for non-cached results
   - Support for 1000+ concurrent users
   - Model training should complete within 30 minutes

---

**Part of the Ghibli Food Recipe Platform Ecosystem** 🍜✨