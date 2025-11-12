import json
from typing import List, Union
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
