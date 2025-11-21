import torch
import insightface
from transformers import BlipProcessor, BlipForConditionalGeneration, CLIPProcessor, CLIPModel
import ssl
import urllib3
import os
import requests
from app.config import DEVICE, FACE_DETECTION_MODEL, IMAGE_DESCRIPTION_MODEL, CLIP_MODEL
from app.utils.logging import get_logger

logger = get_logger(__name__)

# Disable SSL warnings and verification for corporate environments
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
ssl._create_default_https_context = ssl._create_unverified_context

# Patch requests globally to disable SSL verification
original_request = requests.Session.request
def patched_request(self, method, url, **kwargs):
    kwargs['verify'] = False
    return original_request(self, method, url, **kwargs)
requests.Session.request = patched_request

# Global model state
_face_app = None
_blip_processor = None
_blip_model = None
_clip_processor = None
_clip_model = None

def get_face_model():
    """Lazy load InsightFace model"""
    global _face_app
    if _face_app is None:
        logger.info(f"Loading InsightFace model ({FACE_DETECTION_MODEL}) on {DEVICE}")
        try:
            import requests
            session = requests.Session()
            session.verify = False
            
            original_get = requests.get
            def patched_get(*args, **kwargs):
                kwargs['verify'] = False
                return original_get(*args, **kwargs)
            requests.get = patched_get
            
            _face_app = insightface.app.FaceAnalysis(name=FACE_DETECTION_MODEL)
            _face_app.prepare(ctx_id=0 if DEVICE == "cuda" else -1, det_size=(640, 640))
            
            requests.get = original_get
            logger.info("InsightFace model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load InsightFace: {e}")
            raise
    return _face_app

def get_blip_model():
    """Lazy load BLIP model"""
    global _blip_processor, _blip_model
    if _blip_processor is None or _blip_model is None:
        logger.info(f"Loading BLIP model ({IMAGE_DESCRIPTION_MODEL}) on {DEVICE}")
        try:
            _blip_processor = BlipProcessor.from_pretrained(
                IMAGE_DESCRIPTION_MODEL,
                trust_remote_code=True,
                token=False
            )
            _blip_model = BlipForConditionalGeneration.from_pretrained(
                IMAGE_DESCRIPTION_MODEL,
                trust_remote_code=True,
                token=False
            )
            if DEVICE == "cuda":
                _blip_model = _blip_model.to(DEVICE)
            logger.info("BLIP model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load BLIP model: {e}")
            _blip_processor = "FAILED"
            _blip_model = "FAILED"
    return _blip_processor, _blip_model

def get_clip_model():
    """Lazy load CLIP model"""
    global _clip_processor, _clip_model
    if _clip_processor is None or _clip_model is None:
        logger.info(f"Loading CLIP model ({CLIP_MODEL}) on {DEVICE}")
        try:
            _clip_processor = CLIPProcessor.from_pretrained(
                CLIP_MODEL,
                trust_remote_code=True,
                token=False
            )
            _clip_model = CLIPModel.from_pretrained(
                CLIP_MODEL,
                trust_remote_code=True,
                token=False
            )
            if DEVICE == "cuda":
                _clip_model = _clip_model.to(DEVICE)
            logger.info("CLIP model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {e}")
            _clip_processor = "FAILED"
            _clip_model = "FAILED"
    return _clip_processor, _clip_model

def is_face_model_loaded():
    """Check if face model is loaded"""
    return _face_app is not None

def is_blip_model_loaded():
    """Check if BLIP model is loaded"""
    return _blip_model is not None and _blip_model != "FAILED"

def is_clip_model_loaded():
    """Check if CLIP model is loaded"""
    return _clip_model is not None and _clip_model != "FAILED"

def get_model_status():
    """Get detailed model loading status"""
    return {
        "face": "loaded" if _face_app is not None else "not_loaded",
        "blip": "loaded" if (_blip_model is not None and _blip_model != "FAILED") else ("failed" if _blip_model == "FAILED" else "not_loaded"),
        "clip": "loaded" if (_clip_model is not None and _clip_model != "FAILED") else ("failed" if _clip_model == "FAILED" else "not_loaded")
    }
