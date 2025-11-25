import json
import cv2
from PIL import Image
from typing import List, Union, Optional
from app.schemas import FaceBounds

from app.utils.logging import get_logger

logger = get_logger(__name__)

def parse_xmp_regions(xmp_regions: Union[str, dict, None], orientation: int, image_path: str = None, image_width: int = None, image_height: int = None) -> List[FaceBounds]:
    """Parse XMP regions (string or dict) and convert to FaceBounds objects with top-left coordinates"""
    if xmp_regions is None:
        return []
    
    logger.debug(f"Parsing XMP regions for {image_path or 'unknown image'}")
        
    try:
        # Get dimensions from XMP if not provided
        if image_width is None or image_height is None:
            if 'AppliedToDimensions' in xmp_regions:
                dims = xmp_regions['AppliedToDimensions']
                image_width = dims.get('W', image_width)
                image_height = dims.get('H', image_height)
        

        # Extract face regions
        known_faces = []
        if 'RegionList' in xmp_regions:
            logger.debug(f"Processing {len(xmp_regions['RegionList'])} XMP regions")
            for region in xmp_regions['RegionList']:
                # Only process Face type regions
                if region.get('Type') == 'Face' and 'Area' in region and 'Name' in region:
                    area = region['Area']
                    
                    # Ensure we have normalized coordinates
                    if area.get('Unit') == 'normalized':
                        # XMP coordinates are center-based, convert to top-left
                        center_x = float(area['X'])
                        center_y = float(area['Y'])
                        w = float(area['W'])
                        h = float(area['H'])
                        
                        # Convert to top-left coordinates
                        x = center_x - w / 2
                        y = center_y - h / 2
                        
                        # Transform coordinates based on EXIF orientation
                        if orientation != 1:
                            if orientation == 2:  # Flip horizontal
                                x = 1 - x - w
                            elif orientation == 3:  # 180° rotation
                                x, y = 1 - x - w, 1 - y - h
                            elif orientation == 4:  # Flip vertical
                                y = 1 - y - h
                            elif orientation == 5:  # 90° CCW + flip horizontal
                                x, y, w, h = y, x, h, w
                            elif orientation == 6:  # 90° clockwise
                                x, y, w, h = 1 - y - h, x, h, w
                            elif orientation == 7:  # 90° CW + flip horizontal
                                x, y, w, h = 1 - y - h, 1 - x - w, h, w
                            elif orientation == 8:  # 90° counter-clockwise
                                x, y, w, h = y, 1 - x - w, h, w
                        
                        centroid_x = x + w / 2
                        centroid_y = y + h / 2
                        
                        known_face = FaceBounds(
                            name=region['Name'],
                            x=x,
                            y=y,
                            w=w,
                            h=h,
                            centroid=[centroid_x, centroid_y]
                        )
                        known_faces.append(known_face)
        
        logger.debug(f"Parsed {len(known_faces)} face regions from XMP")
        return known_faces
        
    except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
        logger.error(f"Error parsing XMP regions: {e}")
        return []

def convert_to_pixels(known_face: FaceBounds, image_width: int, image_height: int) -> List[float]:
    """Convert normalized top-left coordinates to pixel coordinates"""
    x1 = known_face.x * image_width
    y1 = known_face.y * image_height
    w = known_face.w * image_width
    h = known_face.h * image_height
    
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

def apply_picasa_shrinkage(bbox: List[float]) -> List[float]:
    """Apply standard shrinkage to Picasa face regions
    
    Picasa stores full face regions (including hair and chin) in XMP.
    We need to shrink these to find the core face area that InsightFace detects.
    
    Base Padding (from empirical analysis):
    - Left:   19% shrink (XMP regions are 19% too wide on left)
    - Top:    24% shrink (XMP regions are 24% too tall on top) 
    - Right:  19% shrink (XMP regions are 19% too wide on right)
    - Bottom: 14% shrink (XMP regions are 14% too tall on bottom)
    """
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    
    # Apply shrinkage
    new_x1 = x1 + width * 0.19
    new_y1 = y1 + height * 0.24
    new_x2 = x2 - width * 0.19
    new_y2 = y2 - height * 0.14
    
    return [new_x1, new_y1, new_x2, new_y2]

def calculate_centroid_distance(centroid1: List[float], centroid2: List[float]) -> float:
    """Calculate Euclidean distance between two centroids"""
    return ((centroid1[0] - centroid2[0]) ** 2 + (centroid1[1] - centroid2[1]) ** 2) ** 0.5

def match_known_faces(detected_faces: List[dict], known_faces: Optional[List[FaceBounds]], image_path: str) -> tuple[List[dict], List[FaceBounds]]:
    """Match detected faces with XMP face regions using centroid-based least distance matching
    
    Strategy:
    1. Calculate centroids for all detected faces
    2. For each known face, find the detected face with the closest centroid
    3. Assign names based on minimum distance matches
    
    Returns:
        tuple: (matched_faces, unmatched_input_faces)
    """
    if not known_faces or known_faces is None:
        return detected_faces, []
    
    logger.debug(f"Matching {len(detected_faces)} detected faces with {len(known_faces)} known faces")
    
    # Get image dimensions for normalization
    img = cv2.imread(image_path)
    if img is None:
        logger.error(f"Could not load image for face matching: {image_path}")
        return detected_faces, known_faces
    
    image_height, image_width = img.shape[:2]
    
    # Calculate normalized centroids for detected faces
    for face in detected_faces:
        bbox = face['bbox']  # [x1, y1, x2, y2] in pixels from InsightFace
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        centroid_x = center_x / image_width
        centroid_y = center_y / image_height
        face['centroid'] = [centroid_x, centroid_y]
    
    # Track matches
    matched_input_faces = set()
    unmatched_input_faces = []
    used_detected_faces = set()
    
    # For each known face, find closest detected face by centroid distance
    for idx, known_face in enumerate(known_faces):
        if not known_face.centroid:
            unmatched_input_faces.append(known_face)
            continue
            
        best_match = None
        best_distance = float('inf')
        best_face_idx = None
        
        for face_idx, face in enumerate(detected_faces):
            if face_idx in used_detected_faces:
                continue
                
            distance = calculate_centroid_distance(known_face.centroid, face['centroid'])
            if distance < best_distance:
                best_distance = distance
                best_match = face
                best_face_idx = face_idx
        
        if best_match and best_distance < 0.1:  # Reasonable distance threshold
            # Assign name to best match
            best_match['person_name'] = known_face.name
            best_match['input_face_matched'] = True
            best_match['input_face_match_confidence'] = 1.0 - best_distance  # Convert distance to confidence
            best_match['match_strategy'] = 'centroid_distance'
            best_match['input_bbox'] = convert_to_pixels(known_face, image_width, image_height)
            best_match['input_centroid'] = known_face.centroid  # Store original XMP centroid
            matched_input_faces.add(idx)
            used_detected_faces.add(best_face_idx)
        else:
            unmatched_input_faces.append(known_face)
    
    # Set default values for unmatched detected faces
    for face in detected_faces:
        if 'input_face_matched' not in face:
            face['input_face_matched'] = False
            face['input_face_match_confidence'] = 0.0
            face['match_strategy'] = 'none'
            face['input_bbox'] = None
    
    logger.debug(f"Face matching completed: {len(matched_input_faces)} matches, {len(unmatched_input_faces)} unmatched")
    return detected_faces, unmatched_input_faces
