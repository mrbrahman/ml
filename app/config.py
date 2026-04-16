import os
import torch

# GPU Configuration
_device_mode = os.getenv("DEVICE_MODE", "auto").lower()
if _device_mode == "cpu":
    DEVICE = "cpu"
elif _device_mode == "cuda":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
elif _device_mode == "auto":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
else:
    raise ValueError(f"Invalid DEVICE_MODE '{_device_mode}'. Supported values: auto, cuda, cpu")

# Model Configuration
FACE_DETECTION_MODEL = "buffalo_l"  # InsightFace model
FACE_SIMILARITY_THRESHOLD = 0.75
FACE_MATCH_TOP_K = 5  # Check top 5 matches for cluster consensus
CLUSTER_SUGGESTION_THRESHOLD = 0.6  # Minimum similarity for cluster name suggestions
TEXT_SEARCH_MIN_SCORE = 0.25  # Minimum similarity score for text search results
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

# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Ensure directories exist
os.makedirs(FAISS_INDEX_DIR, exist_ok=True)
