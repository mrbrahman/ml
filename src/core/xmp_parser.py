import json
from typing import List, Dict, Any, Union
from src.schemas.models import XmpFace

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

def convert_xmp_to_api_format(xmp_regions_json: str) -> List[Dict[str, Any]]:
    """Convert XMP regions to API format for easy testing"""
    xmp_faces = parse_xmp_regions(xmp_regions_json)
    return [
        {
            "name": face.name,
            "x": face.x,
            "y": face.y,
            "w": face.w,
            "h": face.h
        }
        for face in xmp_faces
    ]