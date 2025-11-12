import numpy as np
from .storage import get_indices

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
