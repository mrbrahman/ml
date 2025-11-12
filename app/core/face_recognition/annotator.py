import cv2
import os
from pathlib import Path
from typing import List, Optional
from app.schemas import FaceInfo, XmpFace
from .xmp_parser import convert_xmp_to_pixels

def create_enriched_image(image_path: str, faces: List[FaceInfo], output_dir: str = "data/annotated_images", xmp_faces: Optional[List[XmpFace]] = None) -> str:
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
    
    # Get image dimensions for XMP coordinate conversion
    image_height, image_width = img.shape[:2]
    
    # Draw XMP faces first (in blue, semi-transparent)
    if xmp_faces and xmp_faces is not None:
        for xmp_face in xmp_faces:
            # Convert XMP coordinates to pixels
            x1, y1, x2, y2 = [int(coord) for coord in convert_xmp_to_pixels(xmp_face, image_width, image_height)]
            
            # Create overlay for transparency
            overlay = img.copy()
            
            # Draw XMP bounding box in blue
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
            # Prepare XMP label
            xmp_label = f"XMP: {xmp_face.name}"
            
            # Calculate text size
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            (text_width, text_height), _ = cv2.getTextSize(xmp_label, font, font_scale, thickness)
            
            # Draw XMP label background in blue
            cv2.rectangle(overlay, (x1, y1 - text_height - 5), (x1 + text_width, y1), (255, 0, 0), -1)
            
            # Apply transparency (0.6)
            cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
            
            # Draw XMP text
            cv2.putText(img, xmp_label, (x1, y1 - 3), font, font_scale, (255, 255, 255), thickness)
    
    # Draw InsightFace detections (in green, semi-transparent)
    for face in faces:
        x1, y1, x2, y2 = [int(coord) for coord in face.bbox]
        
        # Create overlay for transparency
        overlay = img.copy()
        
        # Draw InsightFace bounding box in green
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Prepare label text
        if face.person_name:
            top_label = face.person_name
        else:
            top_label = f"ID: {face.cluster_id}"
        
        # Prepare pose text
        bottom_label = ""
        if face.pose:
            pitch = face.pose.get('pitch', 0)
            yaw = face.pose.get('yaw', 0)
            roll = face.pose.get('roll', 0)
            bottom_label = f"P:{pitch:.1f} Y:{yaw:.1f} R:{roll:.1f}"
        
        # Calculate text sizes
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        (top_width, top_height), _ = cv2.getTextSize(top_label, font, font_scale, thickness)
        (bottom_width, bottom_height), _ = cv2.getTextSize(bottom_label, font, font_scale, thickness)
        
        # Draw InsightFace label background in green
        cv2.rectangle(overlay, (x1, y1 - top_height - 5), (x1 + top_width, y1), (0, 255, 0), -1)
        
        # Draw bottom label background if pose data exists
        if bottom_label:
            cv2.rectangle(overlay, (x1, y2), (x1 + bottom_width, y2 + bottom_height + 5), (0, 255, 0), -1)
        
        # Apply transparency (0.6)
        cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
        
        # Draw InsightFace text
        cv2.putText(img, top_label, (x1, y1 - 3), font, font_scale, (0, 0, 0), thickness)
        if bottom_label:
            cv2.putText(img, bottom_label, (x1, y2 + bottom_height + 2), font, font_scale, (0, 0, 0), thickness)
    
    # Save the enriched image
    cv2.imwrite(output_path, img)
    
    return output_path