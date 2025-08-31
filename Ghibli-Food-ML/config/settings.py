import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    PROJECT_NAME = "Ghibli Food ML Recommendation System"
    VERSION = "1.0.0"
    API_V1_STR = "/api/v1"
    
    # Database
    POSTGRES_SERVER = os.getenv("POSTGRES_SERVER", "localhost")
    POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
    POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "")
    POSTGRES_DB = os.getenv("POSTGRES_DB", "ghibli_food_db")
    POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
    
    DATABASE_URL = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_SERVER}:{POSTGRES_PORT}/{POSTGRES_DB}"
    
    # API Settings
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", 8001))
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"
    
    # Main API URL (your existing backend)
    MAIN_API_URL = os.getenv("MAIN_API_URL", "http://localhost:5000/api/v1")
    
    # ML Model Settings
    MODEL_PATH = os.getenv("MODEL_PATH", "models/")
    MIN_RATING_THRESHOLD = float(os.getenv("MIN_RATING_THRESHOLD", "3.0"))
    RECOMMENDATION_COUNT = int(os.getenv("RECOMMENDATION_COUNT", "5"))
    
    # CORS
    BACKEND_CORS_ORIGINS = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:8000",
        "http://localhost:8001",
    ]

settings = Settings()