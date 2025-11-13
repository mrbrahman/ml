from pydantic import BaseModel
from typing import List, Optional

class XmpFace(BaseModel):
    name: str
    x: float  # Normalized coordinates (0.0-1.0)
    y: float
    w: float
    h: float

class AnalyzeImageRequest(BaseModel):
    image_id: str
    image_path: str
    save_annotated: Optional[bool] = False
    known_faces: Optional[List[XmpFace]] = None
    xmp_regions: Optional[dict] = None  # Raw XMP regions object from exiftool

class FaceInfo(BaseModel):
    bbox: List[float]  # [x, y, w, h]
    confidence: float
    cluster_id: str
    person_name: Optional[str] = None
    gender: Optional[str] = None  # M=male, F=female
    age: Optional[int] = None
    landmarks: Optional[dict] = None  # Named 5-point landmarks
    pose: Optional[dict] = None  # Named head pose angles
    reference_cluster_id: Optional[str] = None  # ID of matched cluster
    reference_image_ids: Optional[List[str]] = None  # Image IDs of all matched faces
    match_confidence: Optional[float] = None  # Similarity score (0.0-1.0)
    consensus_count: Optional[int] = None  # How many faces agreed on match
    is_new_cluster: bool = False  # True if this created a new cluster
    xmp_matched: Optional[bool] = None  # True if matched with XMP face data
    xmp_match_confidence: Optional[float] = None  # IoU confidence for XMP match

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
