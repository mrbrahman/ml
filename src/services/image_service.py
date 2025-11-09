from src.core.face_detector import detect_faces
from src.core.image_captioner import generate_description
from src.core.clip_encoder import encode_image
from src.data.vector_store import vector_store
from src.data.vector_search import search_visual
from src.schemas.models import FaceInfo, AnalyzeImageResponse
from src.infrastructure.config import MODEL_NAMES

def analyze_image(image_id: str, image_path: str) -> AnalyzeImageResponse:
    """Analyze image for faces and generate description"""
    # Remove existing embeddings for this image first
    vector_store.remove_image_embeddings(image_id)
    
    # Always process image fully (allows for model updates)
    faces_data = detect_faces(image_path)
    
    # Process each face with fresh embeddings
    faces_info = []
    for face_data in faces_data:
        cluster_id, person_name, reference_cluster_id, reference_image_ids, match_confidence, consensus_count, is_new_cluster = vector_store.add_face_embedding(
            image_id, 
            face_data['embedding']
        )
        
        faces_info.append(FaceInfo(
            bbox=face_data['bbox'],
            confidence=face_data['confidence'],
            cluster_id=cluster_id,
            person_name=person_name,
            gender=face_data.get('gender'),
            age=face_data.get('age'),
            landmarks=face_data.get('landmarks'),
            pose=face_data.get('pose'),
            reference_cluster_id=reference_cluster_id,
            reference_image_ids=reference_image_ids,
            match_confidence=match_confidence,
            consensus_count=consensus_count,
            is_new_cluster=is_new_cluster
        ))
    
    # Generate image description
    description = generate_description(image_path)
    
    # Replace visual and text embeddings (removes old ones first)
    clip_embedding = encode_image(image_path)
    vector_store.replace_visual_embedding(image_id, clip_embedding)
    vector_store.replace_text_embedding(image_id, clip_embedding)
    
    return AnalyzeImageResponse(
        image_id=image_id,
        image_path=image_path,
        faces=faces_info,
        description=description,
        models_used={
            "face_detection": MODEL_NAMES["face_detection"],
            "image_description": MODEL_NAMES["image_description"],
            "embeddings": MODEL_NAMES["visual_similarity"]
        }
    )

def get_similar_images(image_path: str, limit: int = 10):
    """Find visually similar images"""
    # Get image embedding
    clip_embedding = encode_image(image_path)
    
    # Search for similar images
    results = search_visual(clip_embedding, k=limit)
    
    return results