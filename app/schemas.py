from pydantic import BaseModel
from typing import List, Optional

class FaceBounds(BaseModel):
    name: str
    x: float  # Normalized coordinates (0.0-1.0)
    y: float
    w: float
    h: float
    centroid: Optional[List[float]] = None  # [x, y] normalized centroid

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
    centroid: Optional[List[float]] = None  # [x, y] normalized centroid

class InputFaceMatch(BaseModel):
    matched: Optional[bool] = None
    name: Optional[str] = None
    confidence: Optional[float] = None  # Distance-based confidence for centroid match
    match_strategy: Optional[str] = None  # 'centroid_distance'
    input_bbox: Optional[List[float]] = None  # Pixel coordinates of original XMP region [x1, y1, x2, y2]
    centroid: Optional[List[float]] = None  # [x, y] normalized centroid

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
    min_score: Optional[float] = None

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

class AnnotateImageRequest(BaseModel):
    image_path: str
    faces: List[FaceInfo]
    unmatched_input_faces: Optional[List[FaceBounds]] = None
    output_dir: Optional[str] = "data/annotated_images"

class AnnotateImageResponse(BaseModel):
    success: bool
    annotated_image_path: str
    message: str
