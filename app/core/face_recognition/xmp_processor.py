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
        # Handle both string and dict inputs
        if isinstance(xmp_regions, str):
            # Handle HTML-encoded quotes
            xmp_regions = xmp_regions.replace('&quot;', '"')
            regions_data = json.loads(xmp_regions)
        else:
            regions_data = xmp_regions
        
        # Get dimensions from XMP if not provided
        if image_width is None or image_height is None:
            if 'AppliedToDimensions' in regions_data:
                dims = regions_data['AppliedToDimensions']
                image_width = dims.get('W', image_width)
                image_height = dims.get('H', image_height)
        

        
        # Extract face regions
        known_faces = []
        if 'RegionList' in regions_data:
            logger.debug(f"Processing {len(regions_data['RegionList'])} XMP regions")
            for region in regions_data['RegionList']:
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
                        
                        known_face = FaceBounds(
                            name=region['Name'],
                            x=x,
                            y=y,
                            w=w,
                            h=h
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

def match_known_faces(detected_faces: List[dict], known_faces: Optional[List[FaceBounds]], image_path: str, containment_threshold: float = 0.7, iou_threshold: float = 0.08) -> tuple[List[dict], List[FaceBounds]]:
    """Match detected faces with XMP face regions and assign names
    
    Returns:
        tuple: (matched_faces, unmatched_input_faces)
    """
    if not known_faces or known_faces is None:
        return detected_faces, []
    
    logger.debug(f"Matching {len(detected_faces)} detected faces with {len(known_faces)} known faces")
    
    # Get image dimensions
    img = cv2.imread(image_path)
    if img is None:
        logger.error(f"Could not load image for face matching: {image_path}")
        return detected_faces, known_faces
    
    image_height, image_width = img.shape[:2]
    
    # Convert known faces to pixel coordinates
    xmp_pixel_faces = []
    for known_face in known_faces:
        pixel_coords = convert_to_pixels(known_face, image_width, image_height)
        xmp_pixel_faces.append({
            'name': known_face.name,
            'bbox': pixel_coords,
            'original': known_face
        })
    
    # Track which input faces were matched
    matched_input_faces = set()
    
    # Match detected faces with XMP faces
    matched_faces = []
    for face in detected_faces:
        best_match = None
        best_score = 0.0
        match_type = "none"
        best_match_idx = -1
        
        for idx, known_face in enumerate(xmp_pixel_faces):
            # Try containment first
            containment = calculate_containment(face['bbox'], known_face['bbox'])
            if containment >= containment_threshold:
                if containment > best_score:
                    best_score = containment
                    best_match = known_face
                    best_match_idx = idx
                    match_type = "containment"
            
            # Fallback to IoU if no good containment match
            elif match_type != "containment":
                iou = calculate_iou(face['bbox'], known_face['bbox'])
                if iou >= iou_threshold and iou > best_score:
                    best_score = iou
                    best_match = known_face
                    best_match_idx = idx
                    match_type = "iou"
        
        # Add XMP match information
        if best_match:
            face['person_name'] = best_match['name']
            face['input_face_matched'] = True
            face['input_face_match_confidence'] = best_score
            matched_input_faces.add(best_match_idx)
            logger.debug(f"Matched detected face to '{best_match['name']}' with {match_type} score {best_score:.3f}")
        else:
            face['input_face_matched'] = False
            face['input_face_match_confidence'] = 0.0
        
        matched_faces.append(face)
    
    # Find unmatched input faces
    unmatched_input_faces = []
    for idx, xmp_face in enumerate(xmp_pixel_faces):
        if idx not in matched_input_faces:
            unmatched_input_faces.append(xmp_face['original'])
    
    logger.debug(f"Face matching completed: {len(matched_input_faces)} matches, {len(unmatched_input_faces)} unmatched")
    return matched_faces, unmatched_input_faces
