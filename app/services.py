from typing import List, Optional
from app.schemas import *
from app.core.face_recognition import manager as faces
from app.core.image_analysis import manager as images, search
from app.utils.logging import get_logger

logger = get_logger(__name__)

def analyze_image(image_id: str, image_path: str, orientation: int, xmp_regions: Optional[dict] = None, save_annotated: bool = False) -> CompositeAnalyzeResponse:
    logger.debug(f"Service: analyze_image called for {image_id}")
    return images.analyze_composite(image_id, image_path, orientation, xmp_regions, save_annotated)

def recognize_faces(image_id: str, image_path: str, orientation: int, save_annotated: bool = False, xmp_regions: Optional[dict] = None) -> FaceRecognitionResponse:
    logger.debug(f"Service: recognize_faces called for {image_id}")
    return faces.recognize(image_id, image_path, orientation, save_annotated, xmp_regions)



def search_by_text(query: str, limit: int = 10):
    logger.debug(f"Service: search_by_text called with query '{query}'")
    return search.by_text(query, limit)

def find_similar_images(image_path: str, limit: int = 10):
    return search.find_similar(image_path, limit)

def get_cluster_info(cluster_id: str = None, person_name: str = None) -> InfoResponse:
    return faces.get_cluster_info(cluster_id, person_name)

def caption_image(image_id: str, image_path: str) -> ImageCaptionResponse:
    logger.debug(f"Service: caption_image called for {image_id}")
    return images.generate_caption(image_id, image_path)

def encode_image(image_id: str, image_path: str) -> ImageEncodeResponse:
    logger.debug(f"Service: encode_image called for {image_id}")
    return images.encode_image(image_id, image_path)
