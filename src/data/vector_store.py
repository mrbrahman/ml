import faiss
import numpy as np
import pickle
import os
from typing import Tuple, Optional
from src.infrastructure.config import FAISS_INDEX_DIR, FACE_INDEX_FILE, VISUAL_INDEX_FILE, TEXT_INDEX_FILE, FACE_SIMILARITY_THRESHOLD
import uuid

class VectorStore:
    def __init__(self):
        self.face_index = None
        self.visual_index = None
        self.text_index = None
        
        # Mappings: FAISS index position -> image_id
        self.face_id_mapping = {}
        self.visual_id_mapping = {}
        self.text_id_mapping = {}
        
        # Face clustering: cluster_id -> list of face indices
        self.face_clusters = {}
        self.face_cluster_names = {}  # cluster_id -> name
        
        self._load_indices()
    
    def _load_indices(self):
        """Load existing FAISS indices or create new ones"""
        face_path = os.path.join(FAISS_INDEX_DIR, FACE_INDEX_FILE)
        visual_path = os.path.join(FAISS_INDEX_DIR, VISUAL_INDEX_FILE)
        text_path = os.path.join(FAISS_INDEX_DIR, TEXT_INDEX_FILE)
        
        # Load face index (512-dim for InsightFace)
        if os.path.exists(face_path):
            self.face_index = faiss.read_index(face_path)
            self._load_mappings("face")
        else:
            self.face_index = faiss.IndexFlatIP(512)  # Inner product for normalized embeddings
        
        # Load visual index (512-dim for CLIP)
        if os.path.exists(visual_path):
            self.visual_index = faiss.read_index(visual_path)
            self._load_mappings("visual")
        else:
            self.visual_index = faiss.IndexFlatIP(512)
        
        # Load text index (512-dim for CLIP)
        if os.path.exists(text_path):
            self.text_index = faiss.read_index(text_path)
            self._load_mappings("text")
        else:
            self.text_index = faiss.IndexFlatIP(512)
    
    def _load_mappings(self, index_type: str):
        """Load ID mappings and cluster info"""
        mapping_file = os.path.join(FAISS_INDEX_DIR, f"{index_type}_mapping.pkl")
        if os.path.exists(mapping_file):
            with open(mapping_file, 'rb') as f:
                data = pickle.load(f)
                if index_type == "face":
                    self.face_id_mapping = data.get('id_mapping', {})
                    self.face_clusters = data.get('clusters', {})
                    self.face_cluster_names = data.get('cluster_names', {})
                elif index_type == "visual":
                    self.visual_id_mapping = data.get('id_mapping', {})
                elif index_type == "text":
                    self.text_id_mapping = data.get('id_mapping', {})
    
    def _save_mappings(self, index_type: str):
        """Save ID mappings and cluster info"""
        mapping_file = os.path.join(FAISS_INDEX_DIR, f"{index_type}_mapping.pkl")
        if index_type == "face":
            data = {
                'id_mapping': self.face_id_mapping,
                'clusters': self.face_clusters,
                'cluster_names': self.face_cluster_names
            }
        elif index_type == "visual":
            data = {'id_mapping': self.visual_id_mapping}
        elif index_type == "text":
            data = {'id_mapping': self.text_id_mapping}
        
        with open(mapping_file, 'wb') as f:
            pickle.dump(data, f)
    
    def add_face_embedding(self, image_id: str, embedding: np.ndarray) -> Tuple[str, Optional[str]]:
        """Add face embedding and return cluster_id and person_name if recognized"""
        embedding = embedding.reshape(1, -1).astype('float32')
        
        # Search for similar faces
        if self.face_index.ntotal > 0:
            scores, indices = self.face_index.search(embedding, k=1)
            if scores[0][0] > FACE_SIMILARITY_THRESHOLD:
                # Found similar face, get cluster
                similar_idx = indices[0][0]
                for cluster_id, face_indices in self.face_clusters.items():
                    if similar_idx in face_indices:
                        # Add to existing cluster
                        new_idx = self.face_index.ntotal
                        self.face_index.add(embedding)
                        self.face_id_mapping[new_idx] = image_id
                        self.face_clusters[cluster_id].append(new_idx)
                        self._save_indices()
                        person_name = self.face_cluster_names.get(cluster_id)
                        return cluster_id, person_name
        
        # Create new cluster
        cluster_id = f"cluster_{uuid.uuid4().hex[:8]}"
        new_idx = self.face_index.ntotal
        self.face_index.add(embedding)
        self.face_id_mapping[new_idx] = image_id
        self.face_clusters[cluster_id] = [new_idx]
        self._save_indices()
        return cluster_id, None
    
    def add_visual_embedding(self, image_id: str, embedding: np.ndarray):
        """Add visual similarity embedding"""
        embedding = embedding.reshape(1, -1).astype('float32')
        idx = self.visual_index.ntotal
        self.visual_index.add(embedding)
        self.visual_id_mapping[idx] = image_id
        self._save_indices()
    
    def add_text_embedding(self, image_id: str, embedding: np.ndarray):
        """Add text-searchable embedding"""
        embedding = embedding.reshape(1, -1).astype('float32')
        idx = self.text_index.ntotal
        self.text_index.add(embedding)
        self.text_id_mapping[idx] = image_id
        self._save_indices()
    
    def name_face_cluster(self, cluster_id: str, name: str):
        """Assign name to face cluster"""
        if cluster_id in self.face_clusters:
            self.face_cluster_names[cluster_id] = name
            self._save_mappings("face")
            return True
        return False
    
    def remove_image_embeddings(self, image_id: str):
        """Remove all embeddings for a specific image_id"""
        removed_count = {'faces': 0, 'visual': 0, 'text': 0}
        
        # Remove face embeddings
        indices_to_remove = []
        for idx, stored_id in self.face_id_mapping.items():
            if stored_id == image_id:
                indices_to_remove.append(idx)
        
        for idx in indices_to_remove:
            del self.face_id_mapping[idx]
            removed_count['faces'] += 1
            # Remove from clusters
            for cluster_id, face_indices in list(self.face_clusters.items()):
                if idx in face_indices:
                    face_indices.remove(idx)
                    if not face_indices:  # Remove empty clusters
                        del self.face_clusters[cluster_id]
                        if cluster_id in self.face_cluster_names:
                            del self.face_cluster_names[cluster_id]
        
        # Remove visual embeddings
        indices_to_remove = []
        for idx, stored_id in self.visual_id_mapping.items():
            if stored_id == image_id:
                indices_to_remove.append(idx)
        for idx in indices_to_remove:
            del self.visual_id_mapping[idx]
            removed_count['visual'] += 1
        
        # Remove text embeddings
        indices_to_remove = []
        for idx, stored_id in self.text_id_mapping.items():
            if stored_id == image_id:
                indices_to_remove.append(idx)
        for idx in indices_to_remove:
            del self.text_id_mapping[idx]
            removed_count['text'] += 1
        
        # Log removal if any embeddings were found
        total_removed = sum(removed_count.values())
        if total_removed > 0:
            print(f"Removed existing embeddings for {image_id}: {removed_count['faces']} faces, {removed_count['visual']} visual, {removed_count['text']} text")
    
    def replace_visual_embedding(self, image_id: str, embedding: np.ndarray):
        """Replace visual embedding for image_id"""
        # Remove existing visual embeddings for this image_id
        indices_to_remove = []
        for idx, stored_id in self.visual_id_mapping.items():
            if stored_id == image_id:
                indices_to_remove.append(idx)
        for idx in indices_to_remove:
            del self.visual_id_mapping[idx]
        
        # Add new embedding
        self.add_visual_embedding(image_id, embedding)
    
    def replace_text_embedding(self, image_id: str, embedding: np.ndarray):
        """Replace text embedding for image_id"""
        # Remove existing text embeddings for this image_id
        indices_to_remove = []
        for idx, stored_id in self.text_id_mapping.items():
            if stored_id == image_id:
                indices_to_remove.append(idx)
        for idx in indices_to_remove:
            del self.text_id_mapping[idx]
        
        # Add new embedding
        self.add_text_embedding(image_id, embedding)
    
    def _save_indices(self):
        """Save all FAISS indices and mappings"""
        faiss.write_index(self.face_index, os.path.join(FAISS_INDEX_DIR, FACE_INDEX_FILE))
        faiss.write_index(self.visual_index, os.path.join(FAISS_INDEX_DIR, VISUAL_INDEX_FILE))
        faiss.write_index(self.text_index, os.path.join(FAISS_INDEX_DIR, TEXT_INDEX_FILE))
        
        self._save_mappings("face")
        self._save_mappings("visual")
        self._save_mappings("text")

# Global vector store instance
vector_store = VectorStore()