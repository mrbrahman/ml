from typing import List, Optional
from app.schemas import *
from app.core.face_recognition import manager as faces
from app.core.image_analysis import manager as images, search

def analyze_image(image_id: str, image_path: str, xmp_faces: Optional[List[XmpFace]] = None, xmp_regions: Optional[dict] = None, save_annotated: bool = False) -> CompositeAnalyzeResponse:
    return images.analyze_composite(image_id, image_path, xmp_faces, xmp_regions, save_annotated)

def recognize_faces(image_id: str, image_path: str, save_annotated: bool = False, xmp_faces: Optional[List[XmpFace]] = None, xmp_regions: Optional[dict] = None) -> FaceRecognitionResponse:
    return faces.recognize(image_id, image_path, save_annotated, xmp_faces, xmp_regions)



def search_by_text(query: str, limit: int = 10):
    return search.by_text(query, limit)

def find_similar_images(image_path: str, limit: int = 10):
    return search.find_similar(image_path, limit)

def get_cluster_info() -> InfoResponse:
    return faces.get_cluster_info()

def caption_image(image_id: str, image_path: str) -> ImageCaptionResponse:
    return images.generate_caption(image_id, image_path)

def encode_image(image_id: str, image_path: str) -> ImageEncodeResponse:
    return images.encode_image(image_id, image_path)
