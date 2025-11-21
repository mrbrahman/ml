import faiss
import numpy as np
import pickle
import os
from typing import Tuple, Optional, List
from app.config import FAISS_INDEX_DIR, FACE_INDEX_FILE

# Global face storage state
_face_index = None  # FAISS index storing 512-dim face embeddings for similarity search
_face_id_mapping = {}  # Maps FAISS index position to image_id: {0: "img_123", 1: "img_456"}
_face_clusters = {}  # Groups face indices by cluster: {"cluster_abc": [0, 5, 12], "cluster_def": [1, 3]}
_face_cluster_names = {}  # Maps cluster IDs to person names: {"cluster_abc": "John Doe", "cluster_def": "Jane Smith"}

def _load_face_index():
    """Load existing face FAISS index or create new one"""
    global _face_index, _face_id_mapping, _face_clusters, _face_cluster_names
    
    face_path = os.path.join(FAISS_INDEX_DIR, FACE_INDEX_FILE)
    
    # Load face index (512-dim for InsightFace)
    if os.path.exists(face_path):
        _face_index = faiss.read_index(face_path)
        _load_face_mappings()
    else:
        _face_index = faiss.IndexFlatIP(512)

def _load_face_mappings():
    """Load face ID mappings and cluster info"""
    global _face_id_mapping, _face_clusters, _face_cluster_names
    
    mapping_file = os.path.join(FAISS_INDEX_DIR, "face_mapping.pkl")
    if os.path.exists(mapping_file):
        with open(mapping_file, 'rb') as f:
            data = pickle.load(f)
            _face_id_mapping = data.get('id_mapping', {})
            _face_clusters = data.get('clusters', {})
            _face_cluster_names = data.get('cluster_names', {})

def _save_face_mappings():
    """Save face ID mappings and cluster info"""
    mapping_file = os.path.join(FAISS_INDEX_DIR, "face_mapping.pkl")
    data = {
        'id_mapping': _face_id_mapping,
        'clusters': _face_clusters,
        'cluster_names': _face_cluster_names
    }
    
    with open(mapping_file, 'wb') as f:
        pickle.dump(data, f)

def _save_face_index():
    """Save face FAISS index and mappings"""
    faiss.write_index(_face_index, os.path.join(FAISS_INDEX_DIR, FACE_INDEX_FILE))
    _save_face_mappings()

# Public interface functions
def get_face_index():
    """Get the FAISS face index"""
    return _face_index

def get_face_id_mapping():
    """Get face ID mapping"""
    return _face_id_mapping

def get_face_clusters():
    """Get face clusters and names"""
    return _face_clusters, _face_cluster_names

def get_cluster_name(cluster_id: str) -> Optional[str]:
    """Get name for a cluster"""
    return _face_cluster_names.get(cluster_id)

def add_face_to_index(embedding: np.ndarray) -> int:
    """Add face embedding to FAISS index and return the index"""
    new_idx = _face_index.ntotal
    _face_index.add(embedding)
    _save_face_index()
    return new_idx

def add_face_to_cluster(cluster_id: str, face_idx: int, image_id: str):
    """Add face to existing cluster"""
    _face_id_mapping[face_idx] = image_id
    _face_clusters[cluster_id].append(face_idx)
    _save_face_mappings()

def create_new_cluster(cluster_id: str, face_idx: int, image_id: str):
    """Create new cluster with face"""
    _face_id_mapping[face_idx] = image_id
    _face_clusters[cluster_id] = [face_idx]
    _save_face_mappings()

def move_face_to_cluster(face_idx: int, from_cluster_id: str, to_cluster_id: str):
    """Move face from one cluster to another"""
    _face_clusters[from_cluster_id].remove(face_idx)
    if not _face_clusters[from_cluster_id]:
        del _face_clusters[from_cluster_id]
        if from_cluster_id in _face_cluster_names:
            del _face_cluster_names[from_cluster_id]
    
    _face_clusters[to_cluster_id].append(face_idx)
    _save_face_index()

def move_face_to_new_cluster(face_idx: int, from_cluster_id: str, new_cluster_id: str, person_name: str):
    """Move face to a new cluster with person name"""
    _face_clusters[from_cluster_id].remove(face_idx)
    if not _face_clusters[from_cluster_id]:
        del _face_clusters[from_cluster_id]
        if from_cluster_id in _face_cluster_names:
            del _face_cluster_names[from_cluster_id]
    
    _face_clusters[new_cluster_id] = [face_idx]
    _face_cluster_names[new_cluster_id] = person_name
    _save_face_index()

def name_face_cluster(cluster_id: str, name: str):
    """Assign name to face cluster"""
    if cluster_id in _face_clusters:
        _face_cluster_names[cluster_id] = name
        _save_face_mappings()
        return True
    return False

def remove_face_embeddings(image_id: str):
    """Remove face embeddings for a specific image_id"""
    # Find indices to remove
    indices_to_remove = [idx for idx, stored_id in _face_id_mapping.items() if stored_id == image_id]
    
    if not indices_to_remove:
        return  # No faces found for this image_id
    
    # Remove face embeddings
    for idx in indices_to_remove:
        del _face_id_mapping[idx]
        # Remove from clusters
        for cluster_id, face_indices in list(_face_clusters.items()):
            if idx in face_indices:
                face_indices.remove(idx)
                if not face_indices:  # Remove empty clusters
                    del _face_clusters[cluster_id]
                    if cluster_id in _face_cluster_names:
                        del _face_cluster_names[cluster_id]
    
    print(f"Removed {len(indices_to_remove)} face embeddings for {image_id}")
    _save_face_mappings()

# Initialize face index on module import
_load_face_index()
