import httpx
import asyncio
from typing import List, Dict, Optional
from config.settings import settings
import logging

logger = logging.getLogger(__name__)

class DataFetcher:
    def __init__(self, base_url: str = None):
        self.base_url = base_url or settings.MAIN_API_URL
        self.timeout = httpx.Timeout(30.0)
    
    async def fetch_all_books(self) -> List[Dict]:
        """Fetch all books from the main API"""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(f"{self.base_url}/books")
                response.raise_for_status()
                data = response.json()
                return data.get('data', [])
        except Exception as e:
            logger.error(f"Error fetching books: {e}")
            return []
    
    async def fetch_user_library_data(self) -> List[Dict]:
        """Fetch all user library data for collaborative filtering"""
        try:
            # Note: This would need authentication in a real scenario
            # For now, we'll return mock data or implement a special endpoint
            # You might need to create a special admin endpoint to fetch all user ratings
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(f"{self.base_url}/admin/all-user-ratings")
                response.raise_for_status()
                data = response.json()
                return data.get('data', [])
        except Exception as e:
            logger.error(f"Error fetching user library data: {e}")
            return []
    
    async def fetch_user_ratings(self, user_id: str) -> List[Dict]:
        """Fetch ratings for a specific user"""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                # This endpoint would need to be implemented in your main API
                response = await client.get(f"{self.base_url}/users/{user_id}/ratings")
                response.raise_for_status()
                data = response.json()
                return data.get('data', [])
        except Exception as e:
            logger.error(f"Error fetching user ratings for {user_id}: {e}")
            return []
    
    async def fetch_book_details(self, book_id: str) -> Optional[Dict]:
        """Fetch details for a specific book"""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(f"{self.base_url}/books/{book_id}")
                response.raise_for_status()
                data = response.json()
                return data.get('data')
        except Exception as e:
            logger.error(f"Error fetching book details for {book_id}: {e}")
            return None
    
    def generate_mock_ratings_data(self, books: List[Dict], num_users: int = 50) -> List[Dict]:
        """Generate mock ratings data for demonstration purposes"""
        import random
        import uuid
        
        ratings = []
        user_ids = [str(uuid.uuid4()) for _ in range(num_users)]
        
        for user_id in user_ids:
            # Each user rates 3-8 random books
            num_books_to_rate = random.randint(3, min(8, len(books)))
            rated_books = random.sample(books, num_books_to_rate)
            
            for book in rated_books:
                # Generate realistic ratings (skewed toward higher ratings)
                rating = random.choices([1, 2, 3, 4, 5], weights=[5, 10, 20, 35, 30])[0]
                ratings.append({
                    'userId': user_id,
                    'bookId': book['id'],
                    'rating': rating
                })
        
        logger.info(f"Generated {len(ratings)} mock ratings for {num_users} users")
        return ratings
    
    async def get_training_data(self, use_mock_ratings: bool = True) -> tuple[List[Dict], List[Dict]]:
        """Get all training data needed for the ML models"""
        books = await self.fetch_all_books()
        
        if use_mock_ratings:
            # Use mock data for demonstration
            ratings = self.generate_mock_ratings_data(books)
        else:
            # Try to fetch real user ratings
            ratings = await self.fetch_user_library_data()
        
        logger.info(f"Retrieved {len(books)} books and {len(ratings)} ratings for training")
        return books, ratings