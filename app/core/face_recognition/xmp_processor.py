import json
import cv2
from typing import List, Union, Optional
from app.schemas import XmpFace

def parse_xmp_regions(xmp_regions: Union[str, dict, None]) -> List[XmpFace]:
    """Parse XMP regions (string or dict) and convert to XmpFace objects"""
    if xmp_regions is None:
        return []
        
    try:
        # Handle both string and dict inputs
        if isinstance(xmp_regions, str):
            # Handle HTML-encoded quotes
            xmp_regions = xmp_regions.replace('&quot;', '"')
            regions_data = json.loads(xmp_regions)
        else:
            regions_data = xmp_regions
        
        # Extract face regions
        xmp_faces = []
        if 'RegionList' in regions_data:
            for region in regions_data['RegionList']:
                # Only process Face type regions
                if region.get('Type') == 'Face' and 'Area' in region and 'Name' in region:
                    area = region['Area']
                    
                    # Ensure we have normalized coordinates
                    if area.get('Unit') == 'normalized':
                        xmp_face = XmpFace(
                            name=region['Name'],
                            x=float(area['X']),
                            y=float(area['Y']),
                            w=float(area['W']),
                            h=float(area['H'])
                        )
                        xmp_faces.append(xmp_face)
        
        return xmp_faces
        
    except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
        print(f"Error parsing XMP regions: {e}")
        return []

def convert_xmp_to_pixels(xmp_face: XmpFace, image_width: int, image_height: int) -> List[float]:
    """Convert normalized XMP coordinates to pixel coordinates"""
    # XMP coordinates are center-based
    center_x = xmp_face.x * image_width
    center_y = xmp_face.y * image_height
    w = xmp_face.w * image_width
    h = xmp_face.h * image_height
    
    # Convert center-based to top-left corner based
    x1 = center_x - w / 2
    y1 = center_y - h / 2
    x2 = x1 + w
    y2 = y1 + h
    
    return [x1, y1, x2, y2]

def calculate_containment(inner_box: List[float], outer_box: List[float], threshold: float = 0.7) -> float:
    """Calculate what percentage of inner_box is contained within outer_box"""
    x1_inner, y1_inner, x2_inner, y2_inner = inner_box
    x1_outer, y1_outer, x2_outer, y2_outer = outer_box
    
    # Calculate intersection
    x_left = max(x1_inner, x1_outer)
    y_top = max(y1_inner, y1_outer)
    x_right = min(x2_inner, x2_outer)
    y_bottom = min(y2_inner, y2_outer)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection = (x_right - x_left) * (y_bottom - y_top)
    inner_area = (x2_inner - x1_inner) * (y2_inner - y1_inner)
    
    return intersection / inner_area if inner_area > 0 else 0.0

def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """Calculate Intersection over Union (IoU) between two bounding boxes"""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Calculate intersection
    x_left = max(x1_1, x1_2)
    y_top = max(y1_1, y1_2)
    x_right = min(x2_1, x2_2)
    y_bottom = min(y2_1, y2_2)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection = (x_right - x_left) * (y_bottom - y_top)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

def match_xmp_faces(detected_faces: List[dict], xmp_faces: Optional[List[XmpFace]], image_path: str, containment_threshold: float = 0.7, iou_threshold: float = 0.08) -> List[dict]:
    """Match detected faces with XMP face regions and assign names"""
    if not xmp_faces or xmp_faces is None:
        return detected_faces
    
    # Get image dimensions
    img = cv2.imread(image_path)
    if img is None:
        return detected_faces
    
    image_height, image_width = img.shape[:2]
    
    # Convert XMP faces to pixel coordinates
    xmp_pixel_faces = []
    for xmp_face in xmp_faces:
        pixel_coords = convert_xmp_to_pixels(xmp_face, image_width, image_height)
        xmp_pixel_faces.append({
            'name': xmp_face.name,
            'bbox': pixel_coords
        })
    
    # Match detected faces with XMP faces
    matched_faces = []
    for face in detected_faces:
        best_match = None
        best_score = 0.0
        match_type = "none"
        
        for xmp_face in xmp_pixel_faces:
            # Try containment first
            containment = calculate_containment(face['bbox'], xmp_face['bbox'])
            if containment >= containment_threshold:
                if containment > best_score:
                    best_score = containment
                    best_match = xmp_face
                    match_type = "containment"
            
            # Fallback to IoU if no good containment match
            elif match_type != "containment":
                iou = calculate_iou(face['bbox'], xmp_face['bbox'])
                if iou >= iou_threshold and iou > best_score:
                    best_score = iou
                    best_match = xmp_face
                    match_type = "iou"
        
        # Add XMP match information
        if best_match:
            face['person_name'] = best_match['name']
            face['xmp_matched'] = True
            face['xmp_match_confidence'] = best_score
        else:
            face['xmp_matched'] = False
            face['xmp_match_confidence'] = 0.0
        
        matched_faces.append(face)
    
    return matched_faces