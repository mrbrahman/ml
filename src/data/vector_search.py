import numpy as np
from typing import List, Tuple
from src.data.vector_store import vector_store

def search_faces(embedding: np.ndarray, threshold: float) -> List[Tuple[str, float]]:
    """Search for similar faces"""
    if vector_store.face_index.ntotal == 0:
        return []
    
    embedding = embedding.reshape(1, -1).astype('float32')
    scores, indices = vector_store.face_index.search(embedding, k=min(10, vector_store.face_index.ntotal))
    
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx != -1 and idx in vector_store.face_id_mapping and score > threshold:
            image_id = vector_store.face_id_mapping[idx]
            results.append((image_id, float(score)))
    
    return results

def search_visual(embedding: np.ndarray, k: int = 10) -> List[Tuple[str, float]]:
    """Search for visually similar images"""
    if vector_store.visual_index.ntotal == 0:
        return []
    
    embedding = embedding.reshape(1, -1).astype('float32')
    scores, indices = vector_store.visual_index.search(embedding, k=min(k, vector_store.visual_index.ntotal))
    
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx != -1 and idx in vector_store.visual_id_mapping:
            image_id = vector_store.visual_id_mapping[idx]
            results.append((image_id, float(score)))
    
    return results

def search_text(embedding: np.ndarray, k: int = 10) -> List[Tuple[str, float]]:
    """Search for similar images using text embedding"""
    if vector_store.text_index.ntotal == 0:
        return []
    
    embedding = embedding.reshape(1, -1).astype('float32')
    scores, indices = vector_store.text_index.search(embedding, k=min(k, vector_store.text_index.ntotal))
    
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx != -1 and idx in vector_store.text_id_mapping:
            image_id = vector_store.text_id_mapping[idx]
            results.append((image_id, float(score)))
    
    return results