import numpy as np
from .storage import get_indices
from . import embeddings
from app.schemas import SearchResult
from app.utils.logging import get_logger

logger = get_logger(__name__)

def search_visual(embedding: np.ndarray, k: int = 10):
    """Search for visually similar images"""
    _visual_index, _text_index, _visual_id_mapping, _text_id_mapping = get_indices()
    
    if _visual_index.ntotal == 0:
        return []
    
    embedding = embedding.reshape(1, -1).astype('float32')
    scores, indices = _visual_index.search(embedding, k=min(k, _visual_index.ntotal))
    
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx != -1 and idx in _visual_id_mapping:
            image_id = _visual_id_mapping[idx]
            results.append((image_id, float(score)))
    
    return results

def search_text(embedding: np.ndarray, k: int = 10):
    """Search for similar images using text embedding"""
    _visual_index, _text_index, _visual_id_mapping, _text_id_mapping = get_indices()
    
    if _text_index.ntotal == 0:
        return []
    
    embedding = embedding.reshape(1, -1).astype('float32')
    scores, indices = _text_index.search(embedding, k=min(k, _text_index.ntotal))
    
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx != -1 and idx in _text_id_mapping:
            image_id = _text_id_mapping[idx]
            results.append((image_id, float(score)))
    
    return results

def by_text(query: str, limit: int = 10):
    """Search for images using text query"""
    logger.debug(f"Text search: '{query}', limit: {limit}")
    text_embedding = embeddings.encode_text(query)
    
    if text_embedding is None:
        return []
    
    results = search_text(text_embedding, k=limit)
    
    search_results = [
        SearchResult(image_id=image_id, score=score)
        for image_id, score in results
    ]
    
    logger.debug(f"Text search found {len(search_results)} results")
    return search_results

def find_similar(image_path: str, limit: int = 10):
    """Find visually similar images"""
    logger.debug(f"Visual similarity search for {image_path}, limit: {limit}")
    clip_embedding = embeddings.encode_image(image_path)
    
    if clip_embedding is None:
        return []
    
    results = search_visual(clip_embedding, k=limit)
    
    search_results = [
        SearchResult(image_id=image_id, score=score)
        for image_id, score in results
    ]
    
    logger.debug(f"Visual search found {len(search_results)} results")
    return search_results
