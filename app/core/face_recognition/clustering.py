import numpy as np
from typing import Tuple, Optional, List
from app.config import FACE_SIMILARITY_THRESHOLD, FACE_MATCH_TOP_K, CLUSTER_SUGGESTION_THRESHOLD
import uuid
from . import storage
from app.utils.logging import get_logger

logger = get_logger(__name__)

def create_cluster_id() -> str:
    """Create a new face cluster ID"""
    
    # We want to make this fast, realistically collision free, but not too lenghty
    # A 'seq' would have been a good solution, but since we have to manage
    # storage of the sequence, that's a bit complicated solution for a simple problem.
    # Hence sticking to creating a UUID, but not using the entire string

    # UUID collision probabilities:
    
    # 8 chars (32 bits): ~4.3B possible values
    #  50.0% collision chance at ~65K clusters
    #   1.0% at ~9000 clusters
    #   0.1% at ~2,900 clusters
    
    # 16 chars (64 bits): ~18.4 quintillion values
    #  50.0% collision chance at ~3B clusters
    #   1.0% at ~430M clusters
    #   0.1% at ~136 million clusters
    
    # Using 16 chars for production safety - can handle millions of clusters before meaningful collision risk
    return f"cluster_{uuid.uuid4().hex[:16]}"

def match_and_cluster_face(image_id: str, embedding: np.ndarray) -> Tuple[str, Optional[str], Optional[str], Optional[List[str]], Optional[float], Optional[int], bool]:
    """Match face embedding against existing clusters and assign to best match or create new cluster. Returns cluster_id, person_name, reference_cluster_id, reference_image_ids, match_confidence, consensus_count, is_new_cluster"""
    logger.debug(f"Clustering face for {image_id}")
    embedding = embedding.reshape(1, -1).astype('float32')
    
    # Multi-candidate face matching with cluster consensus
    face_index = storage.get_face_index()
    if face_index.ntotal > 0:
        k = min(FACE_MATCH_TOP_K, face_index.ntotal)
        scores, indices = face_index.search(embedding, k=k)
        
        # Count matches per cluster above threshold
        cluster_matches = {}
        best_score = 0
        face_clusters = storage.get_face_clusters()[0]
        
        for i in range(k):
            if scores[0][i] > FACE_SIMILARITY_THRESHOLD:
                idx = indices[0][i]
                # Find which cluster this face belongs to
                for cluster_id, face_indices in face_clusters.items():
                    if idx in face_indices:
                        if cluster_id not in cluster_matches:
                            cluster_matches[cluster_id] = []
                        cluster_matches[cluster_id].append((idx, scores[0][i]))
                        
                        if scores[0][i] > best_score:
                            best_score = scores[0][i]
                        break
        
        # Choose cluster with most matches (consensus approach)
        if cluster_matches:
            best_cluster = max(cluster_matches.keys(), key=lambda c: len(cluster_matches[c]))
            reference_image_ids = [storage.get_face_id_mapping().get(idx) for idx, _ in cluster_matches[best_cluster]]
            consensus_count = len(cluster_matches[best_cluster])
            
            # Add to existing cluster
            new_idx = storage.add_face_to_index(embedding)
            storage.add_face_to_cluster(best_cluster, new_idx, image_id)
            person_name = storage.get_cluster_name(best_cluster)
            logger.debug(f"Added face from {image_id} to existing cluster {best_cluster} (consensus: {consensus_count})")
            return best_cluster, person_name, best_cluster, reference_image_ids, best_score, consensus_count, False
    
    # Create new cluster
    cluster_id = create_cluster_id()
    new_idx = storage.add_face_to_index(embedding)
    storage.create_new_cluster(cluster_id, new_idx, image_id)
    logger.debug(f"Created new cluster {cluster_id} for face from {image_id}")
    return cluster_id, None, None, None, None, None, True

def get_cluster_name_suggestions(cluster_id: Optional[str] = None, min_similarity: Optional[float] = None) -> List[dict]:
    """Get name suggestions for unnamed clusters based on nearest named cluster centroids"""
    face_clusters, face_cluster_names = storage.get_face_clusters()
    face_index = storage.get_face_index()
    
    if face_index.ntotal == 0:
        return []
    
    # Separate named and unnamed clusters
    named_clusters = {cid: indices for cid, indices in face_clusters.items() if cid in face_cluster_names}
    unnamed_clusters = {cid: indices for cid, indices in face_clusters.items() if cid not in face_cluster_names}
    
    if not named_clusters or not unnamed_clusters:
        return []
    
    # Filter to specific cluster if provided
    if cluster_id:
        if cluster_id not in unnamed_clusters:
            return []
        unnamed_clusters = {cluster_id: unnamed_clusters[cluster_id]}
    
    # Calculate centroids for named clusters
    named_centroids = {}
    for cid, face_indices in named_clusters.items():
        embeddings = [face_index.reconstruct(idx) for idx in face_indices]
        centroid = np.mean(embeddings, axis=0)
        named_centroids[cid] = centroid
    
    suggestions = []
    
    # For each unnamed cluster, find nearest named cluster
    for cid, face_indices in unnamed_clusters.items():
        embeddings = [face_index.reconstruct(idx) for idx in face_indices]
        unnamed_centroid = np.mean(embeddings, axis=0)
        
        best_match = None
        best_similarity = -1
        
        for named_id, named_centroid in named_centroids.items():
            similarity = np.dot(unnamed_centroid, named_centroid)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = named_id
        
        threshold = min_similarity if min_similarity is not None else CLUSTER_SUGGESTION_THRESHOLD
        if best_match and best_similarity >= threshold:
            suggestions.append({
                "cluster_id": cid,
                "face_count": len(face_indices),
                "suggested_name": face_cluster_names[best_match],
                "similarity_score": float(best_similarity),
                "reference_cluster_id": best_match
            })
    
    return sorted(suggestions, key=lambda x: x["similarity_score"], reverse=True)

def correct_face_assignment(image_id: str, person_name: str) -> Tuple[bool, str, str, str]:
    """Correct face assignment by moving to best cluster for given person name"""
    logger.info(f"Correcting face assignment for {image_id} to '{person_name}'")
    # Find the face index for this image_id
    face_idx = None
    current_cluster_id = None
    
    face_clusters, face_cluster_names = storage.get_face_clusters()
    face_id_mapping = storage.get_face_id_mapping()
    
    for cluster_id, face_indices in face_clusters.items():
        for idx in face_indices:
            if face_id_mapping.get(idx) == image_id:
                face_idx = idx
                current_cluster_id = cluster_id
                break
        if face_idx is not None:
            break
    
    if face_idx is None:
        return False, "", "", "Face not found"
    
    # Check if already correctly assigned
    current_name = face_cluster_names.get(current_cluster_id)
    if current_name == person_name:
        return True, current_cluster_id, "already_correct", f"Face already correctly assigned to {person_name}"
    
    # Find clusters with the target person name
    target_clusters = [cid for cid, name in face_cluster_names.items() if name == person_name]
    
    if target_clusters:
        # Find best matching cluster among those with the correct name
        face_index = storage.get_face_index()
        embedding = face_index.reconstruct(face_idx).reshape(1, -1)
        best_cluster_id = None
        best_score = 0
        
        for cluster_id in target_clusters:
            for idx in face_clusters[cluster_id]:
                other_embedding = face_index.reconstruct(idx).reshape(1, -1)
                score = np.dot(embedding, other_embedding.T)[0][0]
                if score > best_score:
                    best_score = score
                    best_cluster_id = cluster_id
        
        if best_cluster_id:
            # Move to existing cluster
            storage.move_face_to_cluster(face_idx, current_cluster_id, best_cluster_id)
            return True, best_cluster_id, "moved_to_existing", f"Moved to existing cluster for {person_name}"
    
    # Create new cluster for this person
    new_cluster_id = create_cluster_id()
    storage.move_face_to_new_cluster(face_idx, current_cluster_id, new_cluster_id, person_name)
    return True, new_cluster_id, "created_new", f"Created new cluster for {person_name}"
