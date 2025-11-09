import faiss
import numpy as np
import pickle
import os
from typing import Tuple, Optional, List
from src.infrastructure.config import FAISS_INDEX_DIR, FACE_INDEX_FILE, VISUAL_INDEX_FILE, TEXT_INDEX_FILE, FACE_SIMILARITY_THRESHOLD, FACE_MATCH_TOP_K
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
    
    def add_face_embedding(self, image_id: str, embedding: np.ndarray) -> Tuple[str, Optional[str], Optional[str], Optional[List[str]], Optional[float], Optional[int], bool]:
        """Add face embedding and return cluster_id, person_name, reference_cluster_id, reference_image_ids, match_confidence, consensus_count, is_new_cluster"""
        embedding = embedding.reshape(1, -1).astype('float32')
        
        # Multi-candidate face matching with cluster consensus:
        # Instead of just checking the single best match, we evaluate multiple
        # top matches to see if they belong to the same cluster (higher confidence)
        if self.face_index.ntotal > 0:
            k = min(FACE_MATCH_TOP_K, self.face_index.ntotal)
            scores, indices = self.face_index.search(embedding, k=k)
            
            # Count matches per cluster above threshold
            cluster_matches = {}
            best_match_idx = None
            best_score = 0
            
            # Evaluate each of the top-k matches
            for i in range(k):
                if scores[0][i] > FACE_SIMILARITY_THRESHOLD:
                    idx = indices[0][i]
                    # Find which cluster this face belongs to
                    for cluster_id, face_indices in self.face_clusters.items():
                        if idx in face_indices:
                            if cluster_id not in cluster_matches:
                                cluster_matches[cluster_id] = []
                            cluster_matches[cluster_id].append((idx, scores[0][i]))
                            
                            # Track best overall match for reference
                            if scores[0][i] > best_score:
                                best_score = scores[0][i]
                                best_match_idx = idx
                            break
            
            # Choose cluster with most matches (consensus approach)
            # This is more reliable than single-match decisions
            if cluster_matches:
                best_cluster = max(cluster_matches.keys(), key=lambda c: len(cluster_matches[c]))
                reference_image_ids = [self.face_id_mapping.get(idx) for idx, _ in cluster_matches[best_cluster]]
                consensus_count = len(cluster_matches[best_cluster])
                
                # Add to existing cluster
                new_idx = self.face_index.ntotal
                self.face_index.add(embedding)
                self.face_id_mapping[new_idx] = image_id
                self.face_clusters[best_cluster].append(new_idx)
                self._save_indices()
                person_name = self.face_cluster_names.get(best_cluster)
                return best_cluster, person_name, best_cluster, reference_image_ids, best_score, consensus_count, False
        
        # Create new cluster
        cluster_id = f"cluster_{uuid.uuid4().hex[:8]}"
        new_idx = self.face_index.ntotal
        self.face_index.add(embedding)
        self.face_id_mapping[new_idx] = image_id
        self.face_clusters[cluster_id] = [new_idx]
        self._save_indices()
        return cluster_id, None, None, None, None, None, True
    
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
    
    def correct_face_assignment(self, image_id: str, person_name: str) -> Tuple[bool, str, str, str]:
        """Correct face assignment by moving to best cluster for given person name"""
        # Find the face index for this image_id
        face_idx = None
        current_cluster_id = None
        
        for cluster_id, face_indices in self.face_clusters.items():
            for idx in face_indices:
                if self.face_id_mapping.get(idx) == image_id:
                    face_idx = idx
                    current_cluster_id = cluster_id
                    break
            if face_idx is not None:
                break
        
        if face_idx is None:
            return False, "", "", "Face not found"
        
        # Check if already correctly assigned
        current_name = self.face_cluster_names.get(current_cluster_id)
        if current_name == person_name:
            return True, current_cluster_id, "already_correct", f"Face already correctly assigned to {person_name}"
        
        # Find clusters with the target person name
        target_clusters = [cid for cid, name in self.face_cluster_names.items() if name == person_name]
        
        if target_clusters:
            # Find best matching cluster among those with the correct name
            embedding = self.face_index.reconstruct(face_idx).reshape(1, -1)
            best_cluster_id = None
            best_score = 0
            
            for cluster_id in target_clusters:
                for idx in self.face_clusters[cluster_id]:
                    other_embedding = self.face_index.reconstruct(idx).reshape(1, -1)
                    score = np.dot(embedding, other_embedding.T)[0][0]
                    if score > best_score:
                        best_score = score
                        best_cluster_id = cluster_id
            
            if best_cluster_id:
                # Move to existing cluster
                self.face_clusters[current_cluster_id].remove(face_idx)
                if not self.face_clusters[current_cluster_id]:
                    del self.face_clusters[current_cluster_id]
                    if current_cluster_id in self.face_cluster_names:
                        del self.face_cluster_names[current_cluster_id]
                
                self.face_clusters[best_cluster_id].append(face_idx)
                self._save_indices()
                return True, best_cluster_id, "moved_to_existing", f"Moved to existing cluster for {person_name}"
        
        # Create new cluster for this person
        new_cluster_id = f"cluster_{uuid.uuid4().hex[:8]}"
        self.face_clusters[current_cluster_id].remove(face_idx)
        if not self.face_clusters[current_cluster_id]:
            del self.face_clusters[current_cluster_id]
            if current_cluster_id in self.face_cluster_names:
                del self.face_cluster_names[current_cluster_id]
        
        self.face_clusters[new_cluster_id] = [face_idx]
        self.face_cluster_names[new_cluster_id] = person_name
        self._save_indices()
        return True, new_cluster_id, "created_new", f"Created new cluster for {person_name}"
    
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