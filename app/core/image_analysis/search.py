import faiss
import numpy as np
import pickle
import os
from app.config import FAISS_INDEX_DIR, VISUAL_INDEX_FILE, TEXT_INDEX_FILE

# Global FAISS indices (visual and text search only)
_visual_index = None
_text_index = None

# Mappings: FAISS index position -> image_id
_visual_id_mapping = {}
_text_id_mapping = {}

def _load_indices():
    """Load existing FAISS indices or create new ones"""
    global _visual_index, _text_index
    
    visual_path = os.path.join(FAISS_INDEX_DIR, VISUAL_INDEX_FILE)
    text_path = os.path.join(FAISS_INDEX_DIR, TEXT_INDEX_FILE)
    
    # Load visual index (512-dim for CLIP)
    if os.path.exists(visual_path):
        _visual_index = faiss.read_index(visual_path)
        _load_mappings("visual")
    else:
        _visual_index = faiss.IndexFlatIP(512)
    
    # Load text index (512-dim for CLIP)
    if os.path.exists(text_path):
        _text_index = faiss.read_index(text_path)
        _load_mappings("text")
    else:
        _text_index = faiss.IndexFlatIP(512)

def _load_mappings(index_type: str):
    """Load ID mappings"""
    global _visual_id_mapping, _text_id_mapping
    
    mapping_file = os.path.join(FAISS_INDEX_DIR, f"{index_type}_mapping.pkl")
    if os.path.exists(mapping_file):
        with open(mapping_file, 'rb') as f:
            data = pickle.load(f)
            if index_type == "visual":
                _visual_id_mapping = data.get('id_mapping', {})
            elif index_type == "text":
                _text_id_mapping = data.get('id_mapping', {})

def _save_mappings(index_type: str):
    """Save ID mappings"""
    mapping_file = os.path.join(FAISS_INDEX_DIR, f"{index_type}_mapping.pkl")
    if index_type == "visual":
        data = {'id_mapping': _visual_id_mapping}
    elif index_type == "text":
        data = {'id_mapping': _text_id_mapping}
    
    with open(mapping_file, 'wb') as f:
        pickle.dump(data, f)

def _save_indices():
    """Save all FAISS indices and mappings"""
    faiss.write_index(_visual_index, os.path.join(FAISS_INDEX_DIR, VISUAL_INDEX_FILE))
    faiss.write_index(_text_index, os.path.join(FAISS_INDEX_DIR, TEXT_INDEX_FILE))
    
    _save_mappings("visual")
    _save_mappings("text")

def add_visual_embedding(image_id: str, embedding: np.ndarray):
    """Add visual similarity embedding"""
    embedding = embedding.reshape(1, -1).astype('float32')
    idx = _visual_index.ntotal
    _visual_index.add(embedding)
    _visual_id_mapping[idx] = image_id
    _save_indices()

def add_text_embedding(image_id: str, embedding: np.ndarray):
    """Add text-searchable embedding"""
    embedding = embedding.reshape(1, -1).astype('float32')
    idx = _text_index.ntotal
    _text_index.add(embedding)
    _text_id_mapping[idx] = image_id
    _save_indices()

def remove_search_embeddings(image_id: str):
    """Remove visual and text embeddings for a specific image_id"""
    removed_count = {'visual': 0, 'text': 0}
    
    # Remove visual embeddings
    indices_to_remove = []
    for idx, stored_id in _visual_id_mapping.items():
        if stored_id == image_id:
            indices_to_remove.append(idx)
    for idx in indices_to_remove:
        del _visual_id_mapping[idx]
        removed_count['visual'] += 1
    
    # Remove text embeddings
    indices_to_remove = []
    for idx, stored_id in _text_id_mapping.items():
        if stored_id == image_id:
            indices_to_remove.append(idx)
    for idx in indices_to_remove:
        del _text_id_mapping[idx]
        removed_count['text'] += 1
    
    # Log removal if any embeddings were found
    total_removed = sum(removed_count.values())
    if total_removed > 0:
        print(f"Removed search embeddings for {image_id}: {removed_count['visual']} visual, {removed_count['text']} text")

def search_visual(embedding: np.ndarray, k: int = 10):
    """Search for visually similar images"""
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

# Initialize indices on module import
_load_indices()
