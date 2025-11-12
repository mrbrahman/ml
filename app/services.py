import os
import json
from pathlib import Path
from typing import List, Optional
from app.schemas import *
from app.core.face_recognition.detection import detect_faces
from app.core.image_analysis.captioning import generate_description
from app.core.image_analysis.encoding import encode_image, encode_text
from app.core.image_analysis import search
from app.core.face_recognition import storage as face_storage
from app.core.face_recognition import recognition as face_recognition
from app.core.face_recognition.xmp_parser import parse_xmp_regions
from app.core.face_recognition.xmp_matcher import match_xmp_faces
from app.core.face_recognition.annotator import create_enriched_image
from app.config import MODEL_NAMES

def analyze_image(image_id: str, image_path: str, xmp_faces: Optional[List[XmpFace]] = None, xmp_regions: Optional[dict] = None, save_annotated: bool = False) -> AnalyzeImageResponse:
    """Analyze image for faces and generate description"""
    face_storage.remove_face_embeddings(image_id)
    search.remove_search_embeddings(image_id)
    faces_data = detect_faces(image_path)
    
    if xmp_regions and not xmp_faces:
        xmp_faces = parse_xmp_regions(xmp_regions)
    
    if xmp_faces:
        faces_data = match_xmp_faces(faces_data, xmp_faces, image_path)
    
    faces_info = []
    for face_data in faces_data:
        cluster_id, person_name, reference_cluster_id, reference_image_ids, match_confidence, consensus_count, is_new_cluster = face_recognition.add_face_embedding(
            image_id, face_data['embedding']
        )
        
        final_person_name = face_data.get('person_name') or person_name
        
        if face_data.get('xmp_matched') and face_data.get('person_name'):
            face_storage.name_face_cluster(cluster_id, face_data['person_name'])
        
        faces_info.append(FaceInfo(
            bbox=face_data['bbox'],
            confidence=face_data['confidence'],
            cluster_id=cluster_id,
            person_name=final_person_name,
            gender=face_data.get('gender'),
            age=face_data.get('age'),
            landmarks=face_data.get('landmarks'),
            pose=face_data.get('pose'),
            reference_cluster_id=reference_cluster_id,
            reference_image_ids=reference_image_ids,
            match_confidence=match_confidence,
            consensus_count=consensus_count,
            is_new_cluster=is_new_cluster,
            xmp_matched=face_data.get('xmp_matched'),
            xmp_match_confidence=face_data.get('xmp_match_confidence')
        ))
    
    description = generate_description(image_path)
    
    if save_annotated and faces_info:
        create_enriched_image(image_path, faces_info, xmp_faces=xmp_faces)
    
    clip_embedding = encode_image(image_path)
    search.add_visual_embedding(image_id, clip_embedding)
    search.add_text_embedding(image_id, clip_embedding)
    
    return AnalyzeImageResponse(
        image_id=image_id,
        image_path=image_path,
        faces=faces_info,
        description=description,
        models_used={
            "face_detection": MODEL_NAMES["face_detection"],
            "image_captioning": MODEL_NAMES["image_description"]
        }
    )

def recognize_faces(image_id: str, image_path: str, save_annotated: bool = False, xmp_faces: Optional[List[XmpFace]] = None, xmp_regions: Optional[dict] = None) -> FaceRecognitionResponse:
    """Face recognition only"""
    face_storage.remove_face_embeddings(image_id)
    faces_data = detect_faces(image_path)
    
    if xmp_regions and not xmp_faces:
        xmp_faces = parse_xmp_regions(xmp_regions)
    
    if xmp_faces:
        faces_data = match_xmp_faces(faces_data, xmp_faces, image_path)
    
    faces_info = []
    for face_data in faces_data:
        cluster_id, person_name, reference_cluster_id, reference_image_ids, match_confidence, consensus_count, is_new_cluster = face_recognition.add_face_embedding(
            image_id, face_data['embedding']
        )
        
        final_person_name = face_data.get('person_name') or person_name
        
        if face_data.get('xmp_matched') and face_data.get('person_name'):
            face_storage.name_face_cluster(cluster_id, face_data['person_name'])
        
        faces_info.append(FaceInfo(
            bbox=face_data['bbox'],
            confidence=face_data['confidence'],
            cluster_id=cluster_id,
            person_name=final_person_name,
            gender=face_data.get('gender'),
            age=face_data.get('age'),
            landmarks=face_data.get('landmarks'),
            pose=face_data.get('pose'),
            reference_cluster_id=reference_cluster_id,
            reference_image_ids=reference_image_ids,
            match_confidence=match_confidence,
            consensus_count=consensus_count,
            is_new_cluster=is_new_cluster,
            xmp_matched=face_data.get('xmp_matched'),
            xmp_match_confidence=face_data.get('xmp_match_confidence')
        ))
    
    if save_annotated and faces_info:
        create_enriched_image(image_path, faces_info, xmp_faces=xmp_faces)
    
    return FaceRecognitionResponse(
        image_id=image_id,
        image_path=image_path,
        faces=faces_info,
        models_used={"face_detection": MODEL_NAMES["face_detection"]}
    )



def search_by_text(query: str, limit: int = 10):
    """Search for images using text query"""
    text_embedding = encode_text(query)
    results = search.search_text(text_embedding, k=limit)
    
    search_results = [
        SearchResult(image_id=image_id, score=score)
        for image_id, score in results
    ]
    
    return search_results

def find_similar_images(image_path: str, limit: int = 10):
    """Find visually similar images"""
    clip_embedding = encode_image(image_path)
    results = search.search_visual(clip_embedding, k=limit)
    
    search_results = [
        SearchResult(image_id=image_id, score=score)
        for image_id, score in results
    ]
    
    return search_results

def get_cluster_info() -> InfoResponse:
    """Get face cluster information"""
    clusters = []
    named_count = 0
    
    face_clusters, face_cluster_names = face_storage.get_face_clusters()
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