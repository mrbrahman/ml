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