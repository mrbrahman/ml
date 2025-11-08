import torch
import insightface
from transformers import BlipProcessor, BlipForConditionalGeneration, CLIPProcessor, CLIPModel
import ssl
import urllib3
import os
import requests
from src.infrastructure.config import DEVICE, FACE_DETECTION_MODEL, IMAGE_DESCRIPTION_MODEL, CLIP_MODEL

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

class ModelManager:
    def __init__(self):
        self.face_app = None
        self.face_app_training = None
        self.blip_processor = None
        self.blip_model = None
        self.clip_processor = None
        self.clip_model = None
        print(f"ModelManager initialized. Models will load on first use. Device: {DEVICE}")
    
    def get_face_model(self):
        """Lazy load InsightFace model"""
        if self.face_app is None:
            print("Loading InsightFace model...")
            try:
                # Patch requests to disable SSL verification
                import requests
                
                # Create session with SSL disabled
                session = requests.Session()
                session.verify = False
                
                # Monkey patch the requests module used by insightface
                original_get = requests.get
                def patched_get(*args, **kwargs):
                    kwargs['verify'] = False
                    return original_get(*args, **kwargs)
                requests.get = patched_get
                
                self.face_app = insightface.app.FaceAnalysis(name=FACE_DETECTION_MODEL)
                self.face_app.prepare(ctx_id=0 if DEVICE == "cuda" else -1, det_size=(640, 640))
                
                # Restore original requests.get
                requests.get = original_get
                
                print("InsightFace model loaded successfully")
            except Exception as e:
                print(f"Failed to load InsightFace: {e}")
                raise
        return self.face_app
    
    def get_face_model_for_training(self):
        """Lazy load InsightFace model with training-specific settings"""
        if self.face_app_training is None:
            print("Loading InsightFace model for training...")
            try:
                import requests
                
                session = requests.Session()
                session.verify = False
                
                original_get = requests.get
                def patched_get(*args, **kwargs):
                    kwargs['verify'] = False
                    return original_get(*args, **kwargs)
                requests.get = patched_get
                
                self.face_app_training = insightface.app.FaceAnalysis(name=FACE_DETECTION_MODEL)
                # Use more permissive settings for training
                self.face_app_training.prepare(ctx_id=0 if DEVICE == "cuda" else -1, det_size=(320, 320), det_thresh=0.3)
                
                requests.get = original_get
                
                print("InsightFace training model loaded successfully")
            except Exception as e:
                print(f"Failed to load InsightFace for training: {e}")
                raise
        return self.face_app_training
    
    def get_blip_model(self):
        """Lazy load BLIP model with fallback"""
        if self.blip_processor is None or self.blip_model is None:
            print("Loading BLIP model...")
            try:
                self.blip_processor = BlipProcessor.from_pretrained(
                    IMAGE_DESCRIPTION_MODEL,
                    trust_remote_code=True,
                    token=False
                )
                self.blip_model = BlipForConditionalGeneration.from_pretrained(
                    IMAGE_DESCRIPTION_MODEL,
                    trust_remote_code=True,
                    token=False
                )
                if DEVICE == "cuda":
                    self.blip_model = self.blip_model.to(DEVICE)
                print("BLIP model loaded successfully")
            except Exception as e:
                print(f"Failed to load BLIP model: {e}")
                print("Using fallback: returning generic descriptions")
                self.blip_processor = None
                self.blip_model = None
        return self.blip_processor, self.blip_model
    
    def get_clip_model(self):
        """Lazy load CLIP model with fallback"""
        if self.clip_processor is None or self.clip_model is None:
            print("Loading CLIP model...")
            try:
                self.clip_processor = CLIPProcessor.from_pretrained(
                    CLIP_MODEL,
                    trust_remote_code=True,
                    token=False
                )
                self.clip_model = CLIPModel.from_pretrained(
                    CLIP_MODEL,
                    trust_remote_code=True,
                    token=False
                )
                if DEVICE == "cuda":
                    self.clip_model = self.clip_model.to(DEVICE)
                print("CLIP model loaded successfully")
            except Exception as e:
                print(f"Failed to load CLIP model: {e}")
                print("Using fallback: generating random embeddings")
                self.clip_processor = None
                self.clip_model = None
        return self.clip_processor, self.clip_model

# Global model manager instance
model_manager = ModelManager()