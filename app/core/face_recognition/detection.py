import cv2
import numpy as np
from app.core.model_loader import get_face_model
from app.utils.logging import get_logger

logger = get_logger(__name__)

def detect_faces(image_path):
    """Detect faces with full attributes for regular use"""
    face_app = get_face_model()
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    faces = face_app.get(img)
    
    results = []
    for face in faces:
        face_data = {
            'bbox': face.bbox.tolist(),
            'confidence': float(face.det_score),
            'embedding': face.normed_embedding
        }
        
        if hasattr(face, 'gender') and face.gender is not None:
            face_data['gender'] = 'M' if int(face.gender) == 1 else 'F'
        if hasattr(face, 'age') and face.age is not None:
            face_data['age'] = int(face.age)
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
                    'pitch': pose[0],
                    'yaw': pose[1],
                    'roll': pose[2]
                }
        
        results.append(face_data)
    
    logger.debug(f"Face detection completed for {image_path}: {len(results)} faces processed")
    return results



def detect_faces_aggressive(image_path, include_attributes=True):
    """Picasa-like aggressive face detection with quality filtering (UNUSED - kept for reference)"""
    face_app = get_face_model()
    img = cv2.imread(image_path)
    
    # Store original settings
    original_thresh = getattr(face_app.det_model, 'nms_thresh', None)
    original_det_thresh = getattr(face_app.det_model, '_score_thresh', None)
    
    try:
        # Aggressive detection settings
        if hasattr(face_app.det_model, 'nms_thresh'):
            face_app.det_model.nms_thresh = 0.1
        if hasattr(face_app.det_model, '_score_thresh'):
            face_app.det_model._score_thresh = 0.02
        
        # Try multiple approaches
        attempts = [
            img,  # Original
            cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)),  # Enhanced contrast
            cv2.resize(img, (320, 320)),  # Smaller scale
            cv2.resize(img, (800, 800)),  # Larger scale
        ]
        
        faces = []
        for attempt_img in attempts:
            if len(attempt_img.shape) == 2:  # Convert grayscale back to BGR
                attempt_img = cv2.cvtColor(attempt_img, cv2.COLOR_GRAY2BGR)
            
            detected = face_app.get(attempt_img)
            if detected:
                faces.extend(detected)
                break  # Use first successful detection
                
    finally:
        # Restore original settings
        if original_thresh is not None:
            face_app.det_model.nms_thresh = original_thresh
        if original_det_thresh is not None:
            face_app.det_model._score_thresh = original_det_thresh
    
    results = []
    for face in faces:
        face_data = {
            'bbox': face.bbox.tolist(),
            'confidence': float(face.det_score),
            'embedding': face.normed_embedding
        }
        
        # Add attributes only if requested
        if include_attributes:
            if hasattr(face, 'gender') and face.gender is not None:
                face_data['gender'] = 'M' if int(face.gender) == 1 else 'F'
            if hasattr(face, 'age') and face.age is not None:
                face_data['age'] = int(face.age)
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
                        'pitch': pose[0],
                        'yaw': pose[1],
                        'roll': pose[2]
                    }
        
        results.append(face_data)
    
    return results
