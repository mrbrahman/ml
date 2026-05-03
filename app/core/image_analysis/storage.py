import faiss
import numpy as np
import pickle
import os
import tempfile
from app.config import FAISS_INDEX_DIR, VISUAL_INDEX_FILE, TEXT_INDEX_FILE
from app.utils.logging import get_logger

logger = get_logger(__name__)

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
        logger.info(f"Loaded visual index with {_visual_index.ntotal} embeddings")
    else:
        _visual_index = faiss.IndexFlatIP(512)
        logger.info("Created new visual index")
    
    # Load text index (512-dim for CLIP)
    if os.path.exists(text_path):
        _text_index = faiss.read_index(text_path)
        _load_mappings("text")
        logger.info(f"Loaded text index with {_text_index.ntotal} embeddings")
    else:
        _text_index = faiss.IndexFlatIP(512)
        logger.info("Created new text index")

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
    """Save ID mappings using atomic write.

    Writes to a temp file in the same directory, then uses os.replace() to
    atomically swap it into place.  This ensures a crash mid-write leaves the
    previous valid file intact (os.replace is atomic on POSIX).
    """
    mapping_file = os.path.join(FAISS_INDEX_DIR, f"{index_type}_mapping.pkl")
    if index_type == "visual":
        data = {'id_mapping': _visual_id_mapping}
    elif index_type == "text":
        data = {'id_mapping': _text_id_mapping}

    fd, tmp_path = tempfile.mkstemp(dir=FAISS_INDEX_DIR, suffix=".tmp")
    try:
        with os.fdopen(fd, 'wb') as f:
            pickle.dump(data, f)
        os.replace(tmp_path, mapping_file)
    except BaseException:
        # Clean up the temp file on any failure
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise

def _save_indices():
    """Save all FAISS indices and mappings"""
    faiss.write_index(_visual_index, os.path.join(FAISS_INDEX_DIR, VISUAL_INDEX_FILE))
    faiss.write_index(_text_index, os.path.join(FAISS_INDEX_DIR, TEXT_INDEX_FILE))
    
    _save_mappings("visual")
    _save_mappings("text")

def get_indices():
    """Get access to loaded indices and mappings"""
    return _visual_index, _text_index, _visual_id_mapping, _text_id_mapping

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

def remove_search_embeddings(image_id: str) -> dict:
    """Remove visual and text embeddings for a specific image_id.

    Returns a dict with 'visual' and 'text' counts of removed entries.
    """
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
        logger.info(f"Removed search embeddings for {image_id}: {removed_count['visual']} visual, {removed_count['text']} text")

    # Persist updated mappings to disk
    _save_mappings("visual")
    _save_mappings("text")

    return removed_count

# Initialize indices on module import
_load_indices()