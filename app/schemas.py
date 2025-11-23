from pydantic import BaseModel
from typing import List, Optional

class FaceBounds(BaseModel):
    name: str
    x: float  # Normalized coordinates (0.0-1.0)
    y: float
    w: float
    h: float

class AnalyzeImageRequest(BaseModel):
    image_id: str
    image_path: str
    orientation: int  # EXIF orientation value (1-8)
    save_annotated: Optional[bool] = False
    xmp_regions: Optional[dict] = None  # Raw XMP regions object from exiftool

class ClusterMatch(BaseModel):
    cluster_id: str
    name: Optional[str] = None
    confidence: Optional[float] = None  # Similarity score (0.0-1.0)
    consensus_count: Optional[int] = None  # How many faces agreed on match
    reference_image_ids: Optional[List[str]] = None  # Image IDs of matched faces
    is_new_cluster: bool = False

class InputFaceMatch(BaseModel):
    matched: Optional[bool] = None
    name: Optional[str] = None
    confidence: Optional[float] = None  # IoU confidence for geometric match
    match_strategy: Optional[str] = None  # 'containment', 'shrinkage_containment', 'shrinkage_iou', 'none'
    shrunk_bbox: Optional[List[float]] = None  # Pixel coordinates of shrunk bbox when strategy is shrinkage

class FaceInfo(BaseModel):
    bbox: List[float]  # [x, y, w, h]
    confidence: float  # Face detection confidence
    person_name: Optional[str] = None  # Final resolved name
    gender: Optional[str] = None  # M=male, F=female
    age: Optional[int] = None
    landmarks: Optional[dict] = None  # Named 5-point landmarks
    pose: Optional[dict] = None  # Named head pose angles
    cluster: ClusterMatch
    input_face_match: InputFaceMatch
    name_mismatch: Optional[bool] = None  # True if cluster name differs from input name

class NameClusterRequest(BaseModel):
    name: str

class NameClusterResponse(BaseModel):
    success: bool
    message: str

class SearchRequest(BaseModel):
    query: str
    limit: Optional[int] = 10

class SearchResult(BaseModel):
    image_id: str
    score: float

class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]



class FaceRecognitionResponse(BaseModel):
    image_id: str
    image_path: str
    faces: List[FaceInfo]
    unmatched_input_faces: List[FaceBounds]
    models_used: dict

class ClusterInfo(BaseModel):
    cluster_id: str
    name: Optional[str] = None
    face_count: int

class InfoResponse(BaseModel):
    total_clusters: int
    named_clusters: int
    clusters: List[ClusterInfo]

class UpdatePersonNameRequest(BaseModel):
    old_name: str
    new_name: str

class CorrectFaceAssignmentRequest(BaseModel):
    image_id: str
    person_name: str

class CorrectFaceAssignmentResponse(BaseModel):
    success: bool
    message: str
    cluster_id: str
    action_taken: str  # "moved_to_existing", "created_new", "already_correct"

class ImageCaptionRequest(BaseModel):
    image_id: str
    image_path: str

class ImageCaptionResponse(BaseModel):
    image_id: str
    image_path: str
    description: str
    models_used: dict

class ImageEncodeRequest(BaseModel):
    image_id: str
    image_path: str

class ImageEncodeResponse(BaseModel):
    image_id: str
    image_path: str
    embedding_stored: bool
    models_used: dict

class CompositeAnalyzeResponse(BaseModel):
    face_recognition: FaceRecognitionResponse
    image_caption: ImageCaptionResponse
    image_encode: ImageEncodeResponse
