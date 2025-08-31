# Machine Learning Web Projects

This directory contains machine learning components for web applications.

## Table of Contents
- [Ghibli-Food-ML Overview](#ghibli-food-ml)
- [Project Integration](#project-integration)
- [Quick Setup Guide](#quick-setup-guide)
- [API Endpoints](#api-endpoints)
- [Machine Learning Models](#machine-learning-models)
- [Integration Examples](#integration-examples)

---

## Projects

### Ghibli-Food-ML
A comprehensive machine learning recommendation system for the Ghibli Food Recipe application.

**Core Features:**
- **Content-Based Filtering**: TF-IDF vectorization and cosine similarity for book features
- **Collaborative Filtering**: Matrix factorization (SVD) for user-item interactions
- **Hybrid Recommendation**: Combines content and collaborative approaches
- **Real-time API**: FastAPI service with automatic documentation
- **Model Training**: Continuous learning from user interactions
- **Smart Search**: Enhanced search with semantic similarity
- **Behavioral Analysis**: User pattern recognition and preference learning

**Advanced Features:**
- **Cold Start Problem**: Handling new users and books with no interaction history
- **Scalable Architecture**: Designed for high-throughput recommendation serving
- **A/B Testing**: Framework for testing different recommendation algorithms
- **Model Monitoring**: Performance tracking and model drift detection
- **Feature Engineering**: Automated extraction of book and user features

---

## 🔗 Project Integration

This ML service acts as the intelligence layer of the Ghibli Food Recipe platform:

### 🔧 Backend API Integration (Back-End-Web)
- **Recommendation Endpoint**: Serves personalized book recommendations
- **Behavioral Tracking**: Receives user interaction data for model training
- **Search Enhancement**: Provides semantic search capabilities
- **Real-time Updates**: Processes new books and users immediately

### 🎨 Frontend Integration (Front-End-Web)
- **Smart Recommendations**: Powers the recommendation sections
- **Enhanced Search**: Improves search results with ML-driven relevance
- **User Behavior**: Tracks interactions for continuous learning
- **Personalization**: Adapts interface based on user preferences

### 🗄️ Database Integration (Database-Web)
- **Feature Storage**: Caches computed features for fast recommendations
- **Model Persistence**: Stores trained models and user profiles
- **Analytics Data**: Accesses user interaction history and book metadata
- **Performance Metrics**: Logs recommendation accuracy and user feedback

### 🚀 DevOps Integration (DevOps-Web)
- **Container Deployment**: Docker-ready with health checks
- **Model Versioning**: MLOps pipeline for model deployment
- **Monitoring**: Performance and accuracy metrics collection
- **Auto-scaling**: Kubernetes horizontal scaling based on load

---

## 🚀 Quick Setup Guide

### Prerequisites
- **Python 3.11+** with pip
- **PostgreSQL** (for data access)
- **Redis** (for caching - optional)
- **Docker** (for containerization)

### Local Development Setup

1. **Environment Setup**
   ```bash
   cd Machine-Learnimg-Web/Ghibli-Food-ML
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Configuration**
   ```bash
   cp .env.example .env
   ```
   
   Configure the following variables:
   ```env
   # API Configuration
   API_HOST=0.0.0.0
   API_PORT=8001
   
   # Database Connection (matches Database-Web)
   DATABASE_URL=postgresql://ghibli_api_user:password@localhost:5432/ghibli_food_db
   
   # Backend API Integration
   BACKEND_API_URL=http://localhost:5000/api/v1
   
   # Redis Cache (optional)
   REDIS_URL=redis://localhost:6379/0
   
   # Model Configuration
   MODEL_UPDATE_INTERVAL=3600  # Update models every hour
   RECOMMENDATION_CACHE_TTL=1800  # Cache for 30 minutes
   
   # Feature Engineering
   MIN_INTERACTIONS_PER_USER=5
   MIN_INTERACTIONS_PER_BOOK=3
   FEATURE_VECTOR_SIZE=100
   ```

3. **Start the Service**
   ```bash
   python src/main.py
   ```
   
   Access the API documentation at: `http://localhost:8001/docs`

### Integrated Development (Full Stack)

Start all services together:
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

Using the integrated Docker setup:
```bash
cd DevOps-Web/Ghibli-Food-DevOps
docker-compose up -d
```

The ML service will be available at `http://localhost:8001`

---

## 🔌 API Endpoints

### Recommendation Endpoints

**POST** `/recommend` - Get personalized recommendations
```json
{
  "user_id": "uuid",
  "num_recommendations": 10,
  "preferences": {
    "genres": ["cookbook", "asian"],
    "difficulty": "intermediate",
    "exclude_read": true
  }
}
```

**GET** `/similar/{book_id}` - Get similar books
```json
{
  "book_id": "uuid",
  "limit": 5
}
```

### Search Endpoints

**POST** `/search` - Enhanced semantic search
```json
{
  "query": "quick pasta recipes",
  "filters": {
    "genres": ["italian"],
    "max_difficulty": "intermediate"
  },
  "limit": 20
}
```

**GET** `/search/suggestions` - Auto-complete suggestions
```json
{
  "query": "chic",
  "limit": 10
}
```

### Analytics Endpoints

**POST** `/behavior` - Track user behavior
```json
{
  "user_id": "uuid",
  "book_id": "uuid",
  "action": "view|rate|add_to_library|purchase",
  "metadata": {
    "rating": 5,
    "time_spent": 120
  },
  "timestamp": "2024-01-01T00:00:00Z"
}
```

**GET** `/analytics/user/{user_id}` - User analytics
```json
{
  "user_profile": {
    "preferences": {},
    "behavior_patterns": {},
    "recommendation_accuracy": 0.85
  }
}
```

### Model Management Endpoints

**POST** `/models/train` - Trigger model retraining
**GET** `/models/status` - Model status and metrics
**GET** `/health` - Service health check

---

## 🤖 Machine Learning Models

### Content-Based Filtering

**TF-IDF Vectorization**
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class ContentBasedRecommender:
    def __init__(self):
        self.tfidf = TfidfVectorizer(
            max_features=10000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.book_features = None
        self.book_similarity_matrix = None
    
    def fit(self, books_df):
        # Combine title, description, ingredients, and genre
        content = books_df['title'] + ' ' + books_df['description'] + ' ' + \
                 books_df['genre'] + ' ' + books_df['ingredients'].str.join(' ')
        
        self.book_features = self.tfidf.fit_transform(content)
        self.book_similarity_matrix = cosine_similarity(self.book_features)
        
    def get_similar_books(self, book_id, n_recommendations=10):
        book_idx = self.book_id_to_index[book_id]
        similarity_scores = self.book_similarity_matrix[book_idx]
        similar_indices = similarity_scores.argsort()[-n_recommendations-1:-1][::-1]
        return [self.index_to_book_id[idx] for idx in similar_indices]
```

### Collaborative Filtering

**Matrix Factorization with SVD**
```python
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix

class CollaborativeFilteringRecommender:
    def __init__(self, n_components=100):
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        self.user_item_matrix = None
        self.user_factors = None
        self.item_factors = None
    
    def fit(self, interactions_df):
        # Create user-item interaction matrix
        self.user_item_matrix = self._create_interaction_matrix(interactions_df)
        
        # Fit SVD model
        self.user_factors = self.svd.fit_transform(self.user_item_matrix)
        self.item_factors = self.svd.components_.T
    
    def get_user_recommendations(self, user_id, n_recommendations=10):
        user_idx = self.user_id_to_index[user_id]
        user_vector = self.user_factors[user_idx]
        
        # Calculate scores for all items
        scores = np.dot(user_vector, self.item_factors.T)
        
        # Get top recommendations (excluding already interacted items)
        already_interacted = set(self.user_item_matrix[user_idx].nonzero()[1])
        recommendations = []
        
        for item_idx in scores.argsort()[::-1]:
            if item_idx not in already_interacted and len(recommendations) < n_recommendations:
                recommendations.append(self.index_to_book_id[item_idx])
                
        return recommendations
```

### Hybrid Recommendation System

**Weighted Combination Approach**
```python
class HybridRecommender:
    def __init__(self, content_weight=0.4, collaborative_weight=0.6):
        self.content_recommender = ContentBasedRecommender()
        self.collaborative_recommender = CollaborativeFilteringRecommender()
        self.content_weight = content_weight
        self.collaborative_weight = collaborative_weight
    
    def get_recommendations(self, user_id, n_recommendations=10):
        # Get recommendations from both models
        content_recs = self.content_recommender.get_recommendations_for_user(user_id)
        collab_recs = self.collaborative_recommender.get_user_recommendations(user_id)
        
        # Combine scores using weighted approach
        combined_scores = {}
        
        for book_id, score in content_recs.items():
            combined_scores[book_id] = score * self.content_weight
            
        for book_id, score in collab_recs.items():
            if book_id in combined_scores:
                combined_scores[book_id] += score * self.collaborative_weight
            else:
                combined_scores[book_id] = score * self.collaborative_weight
        
        # Return top N recommendations
        sorted_recs = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        return [book_id for book_id, _ in sorted_recs[:n_recommendations]]
```

### Cold Start Handling

**New User Recommendations**
```python
class ColdStartHandler:
    def get_new_user_recommendations(self, user_preferences, n_recommendations=10):
        # Use popularity-based and content-based recommendations
        popular_books = self._get_popular_books_by_genre(user_preferences.get('genres', []))
        
        if user_preferences.get('sample_ratings'):
            # Use sample ratings to find similar users
            similar_users = self._find_similar_users_by_sample_ratings(
                user_preferences['sample_ratings']
            )
            similar_user_books = self._get_books_liked_by_users(similar_users)
            return self._combine_recommendations(popular_books, similar_user_books)
        
        return popular_books[:n_recommendations]
    
    def get_new_book_recommendations(self, book_metadata):
        # Use content-based similarity with existing books
        similar_books = self.content_recommender.find_similar_by_metadata(book_metadata)
        return similar_books
```

---

## 🔌 Integration Examples

### Backend API Integration

**Recommendation Service Integration**
```python
# In backend API (Node.js/Express)
const axios = require('axios')

class MLService {
    constructor() {
        this.baseURL = process.env.ML_SERVICE_URL || 'http://localhost:8001'
    }
    
    async getRecommendations(userId, preferences = {}) {
        try {
            const response = await axios.post(`${this.baseURL}/recommend`, {
                user_id: userId,
                preferences,
                num_recommendations: 10
            })
            
            return response.data
        } catch (error) {
            console.error('ML service error:', error)
            return { recommendations: [], fallback: true }
        }
    }
    
    async trackBehavior(userId, bookId, action, metadata = {}) {
        try {
            await axios.post(`${this.baseURL}/behavior`, {
                user_id: userId,
                book_id: bookId,
                action,
                metadata,
                timestamp: new Date().toISOString()
            })
        } catch (error) {
            console.error('Failed to track behavior:', error)
        }
    }
    
    async searchBooks(query, filters = {}) {
        try {
            const response = await axios.post(`${this.baseURL}/search`, {
                query,
                filters
            })
            
            return response.data
        } catch (error) {
            console.error('Search service error:', error)
            return { results: [], fallback: true }
        }
    }
}

module.exports = new MLService()
```

### Real-time Model Updates

**Continuous Learning Pipeline**
```python
import asyncio
from datetime import datetime, timedelta

class ModelUpdateService:
    def __init__(self):
        self.last_update = datetime.now()
        self.update_interval = timedelta(hours=1)
        self.batch_size = 1000
        
    async def start_update_loop(self):
        while True:
            try:
                await self.check_and_update_models()
                await asyncio.sleep(300)  # Check every 5 minutes
            except Exception as e:
                logger.error(f"Model update error: {e}")
                await asyncio.sleep(600)  # Wait longer on error
    
    async def check_and_update_models(self):
        if datetime.now() - self.last_update > self.update_interval:
            logger.info("Starting model update...")
            
            # Get new interaction data
            new_interactions = await self.get_new_interactions()
            
            if len(new_interactions) > self.batch_size:
                # Retrain models with new data
                await self.retrain_models(new_interactions)
                self.last_update = datetime.now()
                logger.info("Model update completed")
    
    async def retrain_models(self, new_data):
        # Update collaborative filtering model
        await self.update_collaborative_model(new_data)
        
        # Update content-based model if new books added
        new_books = await self.get_new_books()
        if new_books:
            await self.update_content_model(new_books)
        
        # Update hybrid model weights based on performance
        await self.optimize_hybrid_weights()
```

### Performance Monitoring

**Model Performance Tracking**
```python
class ModelMonitor:
    def __init__(self):
        self.metrics = {
            'precision_at_k': [],
            'recall_at_k': [],
            'ndcg': [],
            'response_time': [],
            'recommendation_diversity': []
        }
    
    async def evaluate_recommendations(self, user_id, recommendations, actual_interactions):
        # Calculate precision@k, recall@k, NDCG
        precision_k = self.calculate_precision_at_k(recommendations, actual_interactions, k=10)
        recall_k = self.calculate_recall_at_k(recommendations, actual_interactions, k=10)
        ndcg = self.calculate_ndcg(recommendations, actual_interactions, k=10)
        
        # Track diversity
        diversity = self.calculate_recommendation_diversity(recommendations)
        
        # Store metrics
        self.metrics['precision_at_k'].append(precision_k)
        self.metrics['recall_at_k'].append(recall_k)
        self.metrics['ndcg'].append(ndcg)
        self.metrics['recommendation_diversity'].append(diversity)
        
        # Send metrics to monitoring system
        await self.send_metrics_to_monitoring()
    
    def calculate_precision_at_k(self, recommendations, actual, k):
        recommended_k = recommendations[:k]
        relevant = set(actual)
        return len(set(recommended_k) & relevant) / k
```

**Integration:**
This ML service integrates with the Back-End-Web and Front-End-Web projects to provide intelligent recipe book recommendations based on user preferences and behavior.

## Architecture

```
Machine-Learning-Web/
├── Ghibli-Food-ML/          # ML recommendation service
│   ├── src/                 # Source code
│   ├── api/                 # API endpoints
│   ├── models/              # ML models
│   ├── utils/               # Utilities
│   ├── notebooks/           # Analysis notebooks
│   └── config/              # Configuration
└── README.md                # This file
```

## Getting Started

1. Navigate to the specific project directory
2. Follow the project-specific README instructions
3. Ensure the main API (Back-End-Web) is running for data integration

## Technologies Used

- **Python 3.11+**
- **FastAPI** - Web framework
- **scikit-learn** - Machine learning library
- **pandas & numpy** - Data processing
- **PostgreSQL** - Database
- **Docker** - Containerization
- **Jupyter** - Data analysis

## Contributing

Each project has its own contribution guidelines. Please refer to the individual project README files for specific instructions.