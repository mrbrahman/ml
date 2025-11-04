from pydantic import BaseModel
from typing import List, Optional

class AnalyzeImageRequest(BaseModel):
    image_id: str
    image_path: str

class FaceInfo(BaseModel):
    bbox: List[float]  # [x, y, w, h]
    confidence: float
    cluster_id: str
    person_name: Optional[str] = None
    gender: Optional[str] = None  # M=male, F=female
    age: Optional[int] = None
    landmarks: Optional[dict] = None  # Named 5-point landmarks
    pose: Optional[dict] = None  # Named head pose angles

class AnalyzeImageResponse(BaseModel):
    image_id: str
    image_path: str
    faces: List[FaceInfo]
    description: str
    models_used: dict

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

class TrainRequest(BaseModel):
    dataset_path: str
    dataset_type: str = "directory"  # "directory" or "json"

class TrainResponse(BaseModel):
    success: bool
    message: str
    faces_trained: int

class FaceRecognitionResponse(BaseModel):
    image_id: str
    image_path: str
    faces: List[FaceInfo]
    models_used: dict