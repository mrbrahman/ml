import cv2
import os
from pathlib import Path
from typing import List
from src.schemas.models import FaceInfo

def create_enriched_image(image_path: str, faces: List[FaceInfo], output_dir: str = "data/annotated_images") -> str:
    """Create an enriched image with bounding boxes and face labels"""
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the original image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Get original image filename without directory structure
    original_filename = Path(image_path).name
    output_path = os.path.join(output_dir, original_filename)
    
    # Draw bounding boxes and labels for each face
    for face in faces:
        x1, y1, x2, y2 = [int(coord) for coord in face.bbox]
        
        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Prepare label text
        label_parts = []
        if face.person_name:
            label_parts.append(face.person_name)
        else:
            label_parts.append(f"ID: {face.cluster_id[:8]}")
        
        label = " | ".join(label_parts)
        
        # Calculate text size and position
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        
        # Draw background rectangle for text
        cv2.rectangle(img, (x1, y1 - text_height - 10), (x1 + text_width, y1), (0, 255, 0), -1)
        
        # Draw text
        cv2.putText(img, label, (x1, y1 - 5), font, font_scale, (0, 0, 0), thickness)
    
    # Save the enriched image
    cv2.imwrite(output_path, img)
    
    return output_path