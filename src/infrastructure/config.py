import os
import torch

# GPU Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_GPU = torch.cuda.is_available()

# Model Configuration
FACE_DETECTION_MODEL = "buffalo_l"  # InsightFace model
FACE_SIMILARITY_THRESHOLD = 0.6
IMAGE_DESCRIPTION_MODEL = "Salesforce/blip2-opt-2.7b"
CLIP_MODEL = "openai/clip-vit-base-patch32"

# Model names for API responses
MODEL_NAMES = {
    "face_detection": f"InsightFace {FACE_DETECTION_MODEL}",
    "image_description": IMAGE_DESCRIPTION_MODEL,
    "text_search": CLIP_MODEL,
    "visual_similarity": CLIP_MODEL
}

# FAISS Configuration
FAISS_INDEX_DIR = "data/faiss_indices"
FACE_INDEX_FILE = "faces.index"
VISUAL_INDEX_FILE = "visual.index"
TEXT_INDEX_FILE = "text.index"

# API Configuration
HOST = "0.0.0.0"
PORT = 8000

# Ensure FAISS directory exists
os.makedirs(FAISS_INDEX_DIR, exist_ok=True)