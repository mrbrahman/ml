import torch
import numpy as np
from PIL import Image
from app.core.model_loader import get_clip_model
from app.config import DEVICE

def encode_image(image_path):
    """Get CLIP embedding for text-searchable similarity"""
    clip_processor, clip_model = get_clip_model()
    
    if clip_processor == "FAILED" or clip_model == "FAILED":
        return None
    
    if clip_processor is None or clip_model is None:
        return None
    
    try:
        image = Image.open(image_path).convert('RGB')
        inputs = clip_processor(images=image, return_tensors="pt")
        
        if DEVICE == "cuda":
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        
        with torch.no_grad():
            image_features = clip_model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        return image_features.cpu().numpy().flatten()
    except Exception as e:
        print(f"Error generating CLIP embedding: {e}")
        return None

def encode_text(text):
    """Get CLIP text embedding for search queries"""
    clip_processor, clip_model = get_clip_model()
    
    if clip_processor == "FAILED" or clip_model == "FAILED":
        return None
    
    if clip_processor is None or clip_model is None:
        return None
    
    try:
        inputs = clip_processor(text=[text], return_tensors="pt", padding=True)
        
        if DEVICE == "cuda":
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        
        with torch.no_grad():
            text_features = clip_model.get_text_features(**inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        return text_features.cpu().numpy().flatten()
    except Exception as e:
        print(f"Error generating text embedding: {e}")
        return None