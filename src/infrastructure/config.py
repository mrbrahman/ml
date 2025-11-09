import os
import torch

# GPU Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_GPU = torch.cuda.is_available()

# Model Configuration
FACE_DETECTION_MODEL = "buffalo_l"  # InsightFace model
# Face matching uses multi-candidate evaluation with cluster consensus:
# 1. Search top-k similar faces above threshold (0.75 for accuracy)
# 2. Group matches by cluster, count matches per cluster
# 3. Choose cluster with most matches (consensus approach)
# 4. Return best individual match as reference
FACE_SIMILARITY_THRESHOLD = 0.75
FACE_MATCH_TOP_K = 5  # Check top 5 matches for cluster consensus
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