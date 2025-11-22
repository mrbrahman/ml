from app.schemas import *
from . import captioning, embeddings, storage
from app.config import MODEL_NAMES
from app.utils.logging import get_logger

logger = get_logger(__name__)

def generate_caption(image_id: str, image_path: str) -> ImageCaptionResponse:
    """Generate image caption using BLIP"""
    logger.debug(f"Generating caption for {image_id}")
    description = captioning.generate_description(image_path)
    
    # Check if description indicates model failure
    if "failed to load" in description or "not available" in description or "Error generating" in description:
        model_name = "FAILED"
    else:
        model_name = MODEL_NAMES["image_description"]
    
    return ImageCaptionResponse(
        image_id=image_id,
        image_path=image_path,
        description=description,
        models_used={"image_captioning": model_name}
    )

def encode_image(image_id: str, image_path: str) -> ImageEncodeResponse:
    """Generate and store image embedding using CLIP"""
    logger.debug(f"Encoding image {image_id}")
    storage.remove_search_embeddings(image_id)
    
    clip_embedding = embeddings.encode_image(image_path)
    
    if clip_embedding is not None:
        storage.add_visual_embedding(image_id, clip_embedding)
        storage.add_text_embedding(image_id, clip_embedding)
        embedding_stored = True
        model_name = MODEL_NAMES.get("image_encoding", "clip-vit-base-patch32")
    else:
        embedding_stored = False
        model_name = "FAILED"
    
    return ImageEncodeResponse(
        image_id=image_id,
        image_path=image_path,
        embedding_stored=embedding_stored,
        models_used={"image_encoding": model_name}
    )

def analyze_composite(image_id: str, image_path: str, orientation: int, known_faces=None, xmp_regions=None, save_annotated: bool = False) -> CompositeAnalyzeResponse:
    """Analyze image for faces and generate description by calling individual services"""
    logger.info(f"Starting composite analysis for {image_id}")
    from app.core.face_recognition import manager as face_manager
    
    # Call face recognition service
    face_response = face_manager.recognize(image_id, image_path, orientation, save_annotated, known_faces, xmp_regions)
    
    # Call image captioning service
    caption_response = generate_caption(image_id, image_path)
    
    # Call image encoding service
    encode_response = encode_image(image_id, image_path)
    
    logger.info(f"Composite analysis completed for {image_id}")
    return CompositeAnalyzeResponse(
        face_recognition=face_response,
        image_caption=caption_response,
        image_encode=encode_response
    )