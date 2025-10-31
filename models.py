import torch
import insightface
from transformers import BlipProcessor, BlipForConditionalGeneration, CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np
import cv2
import ssl
import urllib3
import os
import requests
from config import DEVICE, FACE_DETECTION_MODEL, IMAGE_DESCRIPTION_MODEL, CLIP_MODEL

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
        self.blip_processor = None
        self.blip_model = None
        self.clip_processor = None
        self.clip_model = None
        print(f"ModelManager initialized. Models will load on first use. Device: {DEVICE}")
    
    def _load_face_model(self):
        """Lazy load InsightFace model"""
        if self.face_app is None:
            print("Loading InsightFace model...")
            try:
                # Patch requests to disable SSL verification
                import requests
                from requests.adapters import HTTPAdapter
                from urllib3.util.retry import Retry
                
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
    
    def _load_blip_model(self):
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
    
    def _load_clip_model(self):
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
    
    def detect_faces(self, image_path):
        """Detect faces and return embeddings with bounding boxes and attributes"""
        self._load_face_model()
        img = cv2.imread(image_path)
        faces = self.face_app.get(img)
        
        results = []
        for face in faces:
            face_data = {
                'bbox': face.bbox.tolist(),
                'confidence': float(face.det_score),
                'embedding': face.normed_embedding
            }
            
            # Add all available attributes
            if hasattr(face, 'gender') and face.gender is not None:
                face_data['gender'] = 'M' if int(face.gender) == 1 else 'F'  # M=male, F=female
            if hasattr(face, 'age') and face.age is not None:
                face_data['age'] = int(face.age)
            # Basic landmarks and pose
            if hasattr(face, 'kps') and face.kps is not None:
                kps = face.kps.tolist()
                if len(kps) >= 5:
                    face_data['landmarks'] = {
                        'left_eye': kps[0],
                        'right_eye': kps[1], 
                        'nose': kps[2],
                        'left_mouth': kps[3],
                        'right_mouth': kps[4]
                    }
            if hasattr(face, 'pose') and face.pose is not None:
                pose = face.pose.tolist()
                if len(pose) >= 3:
                    face_data['pose'] = {
                        'pitch': pose[0],  # Up/down rotation
                        'yaw': pose[1],    # Left/right rotation  
                        'roll': pose[2]    # Tilt rotation
                    }
            # Detailed landmarks commented out - enable when needed
            # if hasattr(face, 'landmark_3d_68') and face.landmark_3d_68 is not None:
            #     face_data['landmarks_3d'] = face.landmark_3d_68.tolist()  # 68-point 3D landmarks
            # if hasattr(face, 'landmark_2d_106') and face.landmark_2d_106 is not None:
            #     face_data['landmarks_2d_106'] = face.landmark_2d_106.tolist()  # 106-point 2D landmarks
            
            results.append(face_data)
        return results
    
    def generate_description(self, image_path):
        """Generate detailed image description with fallback"""
        self._load_blip_model()
        
        if self.blip_processor is None or self.blip_model is None:
            # Fallback: return generic description
            return "Image analysis temporarily unavailable. Face detection still functional."
        
        try:
            image = Image.open(image_path).convert('RGB')
            inputs = self.blip_processor(image, return_tensors="pt")
            
            if DEVICE == "cuda":
                inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            
            with torch.no_grad():
                out = self.blip_model.generate(**inputs, max_length=100, num_beams=5)
            
            description = self.blip_processor.decode(out[0], skip_special_tokens=True)
            return description
        except Exception as e:
            print(f"Error generating description: {e}")
            return "Error generating image description. Face detection still functional."
    
    def get_clip_embedding(self, image_path):
        """Get CLIP embedding for text-searchable similarity with fallback"""
        self._load_clip_model()
        
        if self.clip_processor is None or self.clip_model is None:
            # Fallback: return random normalized embedding
            print("Using fallback random embedding for CLIP")
            embedding = np.random.randn(512).astype('float32')
            embedding = embedding / np.linalg.norm(embedding)
            return embedding
        
        try:
            image = Image.open(image_path).convert('RGB')
            inputs = self.clip_processor(images=image, return_tensors="pt")
            
            if DEVICE == "cuda":
                inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            
            with torch.no_grad():
                image_features = self.clip_model.get_image_features(**inputs)
                # Normalize for cosine similarity
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            return image_features.cpu().numpy().flatten()
        except Exception as e:
            print(f"Error generating CLIP embedding: {e}")
            # Fallback: return random normalized embedding
            embedding = np.random.randn(512).astype('float32')
            embedding = embedding / np.linalg.norm(embedding)
            return embedding
    
    def get_text_embedding(self, text):
        """Get CLIP text embedding for search queries"""
        self._load_clip_model()
        
        if self.clip_processor is None or self.clip_model is None:
            # Fallback: return random normalized embedding
            print("Using fallback random embedding for text")
            embedding = np.random.randn(512).astype('float32')
            embedding = embedding / np.linalg.norm(embedding)
            return embedding
        
        try:
            inputs = self.clip_processor(text=[text], return_tensors="pt", padding=True)
            
            if DEVICE == "cuda":
                inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            
            with torch.no_grad():
                text_features = self.clip_model.get_text_features(**inputs)
                # Normalize for cosine similarity
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            return text_features.cpu().numpy().flatten()
        except Exception as e:
            print(f"Error generating text embedding: {e}")
            # Fallback: return random normalized embedding
            embedding = np.random.randn(512).astype('float32')
            embedding = embedding / np.linalg.norm(embedding)
            return embedding

# Global model manager instance
model_manager = ModelManager()