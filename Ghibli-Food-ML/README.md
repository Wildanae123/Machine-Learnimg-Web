# Ghibli Food ML Recommendation System

A machine learning-powered recommendation system for the Ghibli Food Recipe application. This service provides personalized book recommendations using both content-based and collaborative filtering approaches.

## 🎯 Features

- **Content-Based Filtering**: Recommends books based on similarity in genres, cuisine types, difficulty levels, and descriptions
- **Collaborative Filtering**: Recommends books based on user rating patterns using matrix factorization (SVD)
- **Hybrid Recommendations**: Combines both approaches for better accuracy
- **RESTful API**: FastAPI-based service with comprehensive endpoints
- **Real-time Training**: Ability to retrain models with new data
- **Docker Support**: Containerized deployment ready
- **Analytics**: Built-in statistics and performance metrics

## 🏗️ Architecture

```
Ghibli-Food-ML/
├── src/
│   ├── main.py                     # FastAPI application entry point
│   └── models/
│       └── recommendation_engine.py # Core ML recommendation algorithms
├── api/
│   └── routes.py                   # API endpoints and request handling
├── config/
│   └── settings.py                 # Configuration and environment variables
├── utils/
│   └── data_fetcher.py            # Data fetching from main API
├── notebooks/
│   └── recommendation_analysis.ipynb # Jupyter notebook for analysis
├── models/                         # Trained model storage
├── data/                          # Data storage
├── requirements.txt               # Python dependencies
├── Dockerfile                     # Container configuration
└── docker-compose.yml            # Multi-service orchestration
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- PostgreSQL (for data storage)
- Your main Ghibli Food API running on `http://localhost:5000`

### Installation

1. **Clone and navigate to the project**:
   ```bash
   cd Machine-Learning-Web/Ghibli-Food-ML
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

4. **Run the service**:
   ```bash
   python src/main.py
   ```

The service will be available at `http://localhost:8001`

### Docker Deployment

1. **Build and run with Docker Compose**:
   ```bash
   docker-compose up -d
   ```

2. **Access the service**:
   - API: `http://localhost:8001`
   - API Documentation: `http://localhost:8001/docs`

## 📚 API Endpoints

### Health & Training

- `GET /health` - Service health check
- `GET /api/v1/recommendations/health` - Recommendation engine health
- `POST /api/v1/recommendations/train` - Train/retrain models

### Recommendations

- `POST /api/v1/recommendations/user` - Get personalized user recommendations
- `POST /api/v1/recommendations/book` - Get similar book recommendations
- `GET /api/v1/recommendations/popular` - Get popular books
- `GET /api/v1/recommendations/trending` - Get trending books
- `GET /api/v1/recommendations/stats` - Get system statistics

### Example API Usage

```python
import httpx

# Train the models (do this first)
async with httpx.AsyncClient() as client:
    response = await client.post(
        "http://localhost:8001/api/v1/recommendations/train",
        json={"use_mock_data": True}
    )

# Get user recommendations
async with httpx.AsyncClient() as client:
    response = await client.post(
        "http://localhost:8001/api/v1/recommendations/user",
        json={
            "user_id": "user123",
            "liked_book_ids": ["book1", "book2"],
            "n_recommendations": 5,
            "recommendation_type": "hybrid"
        }
    )
```

## 🤖 Machine Learning Models

### Content-Based Filtering

- **Algorithm**: TF-IDF Vectorization + Cosine Similarity
- **Features**: Title, description, genre, cuisine type, dietary category, difficulty level, ingredients
- **Use Case**: Recommends books similar to ones the user has liked

### Collaborative Filtering

- **Algorithm**: Matrix Factorization using Truncated SVD
- **Features**: User-book rating matrix
- **Use Case**: Recommends books based on similar users' preferences

### Hybrid Approach

- **Method**: Weighted combination of both approaches
- **Default Weights**: 60% content-based, 40% collaborative
- **Benefits**: Handles cold start problems and improves recommendation diversity

## 🔧 Configuration

Key environment variables in `.env`:

```bash
# API Configuration
HOST=0.0.0.0
PORT=8001
MAIN_API_URL=http://localhost:5000/api/v1

# Database
POSTGRES_SERVER=localhost
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_password
POSTGRES_DB=ghibli_food_db

# ML Settings
MODEL_PATH=models/
MIN_RATING_THRESHOLD=3.0
RECOMMENDATION_COUNT=5
```

## 📊 Data Analysis

Use the included Jupyter notebook for data exploration and model analysis:

```bash
cd notebooks
jupyter notebook recommendation_analysis.ipynb
```

The notebook includes:
- Data visualization and exploration
- Model training and evaluation
- Performance metrics analysis
- Recommendation quality assessment

## 🔄 Integration with Main API

This ML service is designed to complement your existing Ghibli Food API:

1. **Data Source**: Fetches book and rating data from your main API
2. **Recommendations**: Provides recommendations back to your frontend
3. **Real-time**: Can retrain models when new books or ratings are added

### Frontend Integration Example

```javascript
// In your React frontend
const getRecommendations = async (userId, likedBooks) => {
  const response = await fetch('http://localhost:8001/api/v1/recommendations/user', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      user_id: userId,
      liked_book_ids: likedBooks,
      n_recommendations: 6,
      recommendation_type: 'hybrid'
    })
  });
  return response.json();
};
```

## 📈 Performance & Monitoring

### Key Metrics

- **Model Accuracy**: Measured through cross-validation
- **Response Time**: API endpoint latency
- **Coverage**: Percentage of items that can be recommended
- **Diversity**: Variety in recommendation categories

### Monitoring Endpoints

- `/health` - Overall service health
- `/api/v1/recommendations/stats` - Detailed statistics
- `/api/v1/recommendations/health` - Model-specific health

## 🛠️ Development

### Adding New Features

1. **New Recommendation Types**: Extend `RecommendationEngine` class
2. **Custom Features**: Modify feature extraction in `prepare_content_features`
3. **New Endpoints**: Add routes in `api/routes.py`

### Testing

```bash
# Run tests (when implemented)
python -m pytest tests/

# Test API endpoints
curl http://localhost:8001/health
```

## 🚀 Production Deployment

### Docker Production Setup

1. **Build production image**:
   ```bash
   docker build -t ghibli-food-ml:prod .
   ```

2. **Deploy with environment-specific configuration**:
   ```bash
   docker run -d \
     -p 8001:8001 \
     -e DEBUG=false \
     -e MAIN_API_URL=https://your-api.com/api/v1 \
     ghibli-food-ml:prod
   ```

### Scaling Considerations

- **Model Caching**: Models are loaded in memory for fast inference
- **Background Training**: Model retraining happens in background tasks
- **Horizontal Scaling**: Multiple instances can run behind a load balancer

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/new-recommendation-type`
3. Make your changes and test thoroughly
4. Submit a pull request with a clear description

## 📝 License

This project is licensed under the ISC License - see the LICENSE file for details.

## 🔗 Related Projects

- **Back-End-Web**: Main API server for the Ghibli Food application
- **Front-End-Web**: React frontend for the Ghibli Food application

## 📞 Support

For issues and questions:
- Create an issue in the repository
- Check the API documentation at `/docs` endpoint
- Review the Jupyter notebook for usage examples

---

**Built with ❤️ for the Ghibli Food Recipe community**