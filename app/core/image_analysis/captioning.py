import torch
from PIL import Image
from app.core.model_loader import get_blip_model
from app.config import DEVICE
from app.utils.logging import get_logger

logger = get_logger(__name__)

def generate_description(image_path):
    """Generate detailed image description"""
    logger.debug(f"Generating description for {image_path}")
    blip_processor, blip_model = get_blip_model()
    
    if blip_processor == "FAILED" or blip_model == "FAILED":
        return "BLIP model failed to load - image captioning unavailable"
    
    if blip_processor is None or blip_model is None:
        return "BLIP model not available - image captioning unavailable"
    
    try:
        image = Image.open(image_path).convert('RGB')
        inputs = blip_processor(image, return_tensors="pt")
        
        if DEVICE == "cuda":
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        
        with torch.no_grad():
            out = blip_model.generate(**inputs, max_length=100, num_beams=5)
        
        description = blip_processor.decode(out[0], skip_special_tokens=True)
        logger.debug(f"Generated description: {description[:50]}...")
        return description
    except Exception as e:
        logger.error(f"Error generating description: {e}")
        return f"Error generating image description: {e}"
