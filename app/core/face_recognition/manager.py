from typing import List, Optional
from app.schemas import *
from .detection import detect_faces
from .xmp_processor import parse_xmp_regions, match_known_faces
from .clustering import add_face_embedding, correct_face_assignment
from .annotator import create_enriched_image
from . import storage
from app.config import MODEL_NAMES

def recognize(image_id: str, image_path: str, save_annotated: bool = False, known_faces: Optional[List[FaceBounds]] = None, xmp_regions: Optional[dict] = None) -> FaceRecognitionResponse:
    """Face recognition only"""
    storage.remove_face_embeddings(image_id)
    faces_data = detect_faces(image_path)
    
    if xmp_regions and not known_faces:
        known_faces = parse_xmp_regions(xmp_regions, image_path)
    
    unmatched_input_faces = []
    if known_faces:
        faces_data, unmatched_input_faces = match_known_faces(faces_data, known_faces, image_path)
    
    faces_info = []
    for face_data in faces_data:
        cluster_id, person_name, reference_cluster_id, reference_image_ids, match_confidence, consensus_count, is_new_cluster = add_face_embedding(
            image_id, face_data['embedding']
        )
        
        input_name = face_data.get('person_name')
        cluster_name = person_name
        
        # Check for name mismatch
        name_mismatch = None
        if input_name and cluster_name and input_name != cluster_name:
            name_mismatch = True
        elif input_name and cluster_name:
            name_mismatch = False
        
        final_person_name = input_name or cluster_name
        
        if face_data.get('input_face_matched') and input_name:
            storage.name_face_cluster(cluster_id, input_name)
        
        faces_info.append(FaceInfo(
            bbox=face_data['bbox'],
            confidence=face_data['confidence'],
            person_name=final_person_name,
            gender=face_data.get('gender'),
            age=face_data.get('age'),
            landmarks=face_data.get('landmarks'),
            pose=face_data.get('pose'),
            cluster=ClusterMatch(
                cluster_id=cluster_id,
                name=cluster_name,
                confidence=match_confidence,
                consensus_count=consensus_count,
                reference_image_ids=reference_image_ids,
                is_new_cluster=is_new_cluster
            ),
            input_face_match=InputFaceMatch(
                matched=face_data.get('input_face_matched'),
                name=input_name,
                confidence=face_data.get('input_face_match_confidence')
            ),
            name_mismatch=name_mismatch
        ))
    
    if save_annotated and faces_info:
        create_enriched_image(image_path, faces_info, known_faces=known_faces)
    

    
    return FaceRecognitionResponse(
        image_id=image_id,
        image_path=image_path,
        faces=faces_info,
        unmatched_input_faces=unmatched_input_faces,
        models_used={"face_detection": MODEL_NAMES["face_detection"]}
    )

def get_cluster_info() -> InfoResponse:
    """Get face cluster information"""
    clusters = []
    named_count = 0
    
    face_clusters, face_cluster_names = storage.get_face_clusters()
    for cluster_id, face_indices in face_clusters.items():
        name = face_cluster_names.get(cluster_id)
        if name:
            named_count += 1
        
        clusters.append(ClusterInfo(
            cluster_id=cluster_id,
            name=name,
            face_count=len(face_indices)
        ))
    
    return InfoResponse(
        total_clusters=len(clusters),
        named_clusters=named_count,
        clusters=clusters
    )

def correct_assignment(image_id: str, person_name: str) -> CorrectFaceAssignmentResponse:
    """Correct face assignment by moving to best cluster for given person name"""
    success, cluster_id, action_taken, message = correct_face_assignment(image_id, person_name)
    
    return CorrectFaceAssignmentResponse(
        success=success,
        message=message,
        cluster_id=cluster_id,
        action_taken=action_taken
    )

def name_cluster(cluster_id: str, name: str) -> NameClusterResponse:
    """Assign name to face cluster"""
    success = storage.name_face_cluster(cluster_id, name)
    
    if success:
        return NameClusterResponse(
            success=True,
            message=f"Named cluster {cluster_id} as '{name}'"
        )
    else:
        return NameClusterResponse(
            success=False,
            message=f"Cluster {cluster_id} not found"
        )