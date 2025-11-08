import cv2
from src.infrastructure.model_manager import model_manager

def detect_faces(image_path):
    """Detect faces and return embeddings with bounding boxes and attributes"""
    face_app = model_manager.get_face_model()
    img = cv2.imread(image_path)
    faces = face_app.get(img)
    
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
        
        results.append(face_data)
    return results

def detect_faces_for_training(image_path):
    """Detect faces with lower thresholds for training - consumes more samples including blurry ones"""
    face_app = model_manager.get_face_model_for_training()
    img = cv2.imread(image_path)
    
    # Use training model with lower thresholds
    faces = face_app.get(img, max_num=1)
    
    # If no faces found, try with enhanced preprocessing
    if not faces:
        # Try histogram equalization to improve contrast for blurry images
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        enhanced = cv2.equalizeHist(gray)
        enhanced_img = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        faces = face_app.get(enhanced_img, max_num=1)
    
    results = []
    for face in faces:
        face_data = {
            'bbox': face.bbox.tolist(),
            'confidence': float(face.det_score),
            'embedding': face.normed_embedding
        }
        results.append(face_data)
    
    return results