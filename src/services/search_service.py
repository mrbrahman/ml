from src.core.clip_encoder import encode_text, encode_image
from src.data.vector_search import search_text, search_visual
from src.schemas.models import SearchResult

def search_by_text(query: str, limit: int = 10):
    """Search for images using text query"""
    # Get text embedding
    text_embedding = encode_text(query)
    
    # Search for similar images
    results = search_text(text_embedding, k=limit)
    
    search_results = [
        SearchResult(image_id=image_id, score=score)
        for image_id, score in results
    ]
    
    return search_results

def find_similar_images(image_path: str, limit: int = 10):
    """Find visually similar images"""
    # Get image embedding
    clip_embedding = encode_image(image_path)
    
    # Search for similar images
    results = search_visual(clip_embedding, k=limit)
    
    search_results = [
        SearchResult(image_id=image_id, score=score)
        for image_id, score in results
    ]
    
    return search_results