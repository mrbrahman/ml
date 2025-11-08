from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import torch
from src.schemas.models import (
    AnalyzeImageRequest, AnalyzeImageResponse, NameClusterRequest, 
    NameClusterResponse, SearchRequest, SearchResponse, TrainRequest, 
    TrainResponse, FaceRecognitionResponse, InfoResponse, UpdatePersonNameRequest,
    CorrectFaceAssignmentRequest, CorrectFaceAssignmentResponse
)
from src.services.image_service import analyze_image, get_similar_images
from src.services.face_service import recognize_faces, assign_name_to_cluster, train_from_dataset, get_cluster_info, update_cluster_name_by_old_name, correct_face_assignment
from src.services.search_service import search_by_text, find_similar_images
from src.infrastructure.model_manager import model_manager

app = FastAPI(title="AI Photo Analysis Service", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/analyze", response_model=AnalyzeImageResponse)
async def analyze_image_endpoint(request: AnalyzeImageRequest):
    """Analyze image for faces and generate description"""
    
    # Validate image path exists
    if not os.path.exists(request.image_path):
        raise HTTPException(status_code=404, detail="Image file not found")
    
    try:
        return analyze_image(request.image_id, request.image_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/faces/recognize", response_model=FaceRecognitionResponse)
async def recognize_faces_endpoint(request: AnalyzeImageRequest):
    """Face recognition only - detect and identify faces without image description"""
    
    if not os.path.exists(request.image_path):
        raise HTTPException(status_code=404, detail="Image file not found")
    
    try:
        return recognize_faces(request.image_id, request.image_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Face recognition failed: {str(e)}")

@app.put("/faces/{cluster_id}", response_model=NameClusterResponse)
async def name_face_cluster_endpoint(cluster_id: str, request: NameClusterRequest):
    """Assign name to a face cluster"""
    
    success = assign_name_to_cluster(cluster_id, request.name)
    
    if success:
        return NameClusterResponse(
            success=True,
            message=f"Cluster {cluster_id} named as '{request.name}'"
        )
    else:
        return NameClusterResponse(
            success=False,
            message=f"Cluster {cluster_id} not found"
        )

@app.post("/train", response_model=TrainResponse)
async def train_faces_endpoint(request: TrainRequest):
    """Train face recognition from dataset"""
    try:
        success = train_from_dataset(request.dataset_path, request.dataset_type)
        
        if success:
            return TrainResponse(
                success=True,
                message=f"Training completed from {request.dataset_path}",
                faces_trained=0  # Could track this if needed
            )
        else:
            return TrainResponse(
                success=False,
                message=f"Training failed from {request.dataset_path}",
                faces_trained=0
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@app.post("/search/text", response_model=SearchResponse)
async def search_by_text_endpoint(request: SearchRequest):
    """Search for images using text query"""
    try:
        search_results = search_by_text(request.query, request.limit)
        
        return SearchResponse(
            query=request.query,
            results=search_results
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.post("/search/similar", response_model=SearchResponse)
async def search_similar_images_endpoint(request: AnalyzeImageRequest):
    """Find visually similar images"""
    if not os.path.exists(request.image_path):
        raise HTTPException(status_code=404, detail="Image file not found")
    
    try:
        search_results = find_similar_images(request.image_path, limit=10)
        
        return SearchResponse(
            query=f"Similar to {request.image_path}",
            results=search_results
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.post("/faces/update-name", response_model=NameClusterResponse)
async def update_face_cluster_name_endpoint(request: UpdatePersonNameRequest):
    """Update name of an existing face cluster"""
    
    success, cluster_count, face_count = update_cluster_name_by_old_name(request.old_name, request.new_name)
    
    if success:
        return NameClusterResponse(
            success=True,
            message=f"Updated '{request.old_name}' to '{request.new_name}' ({cluster_count} clusters, {face_count} faces)"
        )
    else:
        return NameClusterResponse(
            success=False,
            message=f"Person '{request.old_name}' not found"
        )

@app.post("/faces/correct", response_model=CorrectFaceAssignmentResponse)
async def correct_face_assignment_endpoint(request: CorrectFaceAssignmentRequest):
    """Correct face assignment by providing the correct person name"""
    
    try:
        return correct_face_assignment(request.image_id, request.person_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Face correction failed: {str(e)}")

@app.get("/faceinfo", response_model=InfoResponse)
async def get_face_info():
    """Get face cluster information"""
    return get_cluster_info()

@app.get("/health")
async def health_check():
    """Health check endpoint with detailed GPU information"""
    try:
        cuda_available = torch.cuda.is_available()
        device_info = "cuda" if cuda_available else "cpu"
        
        gpu_info = {}
        if cuda_available:
            try:
                gpu_info = {
                    "gpu_count": torch.cuda.device_count(),
                    "current_device": torch.cuda.current_device(),
                    "device_name": torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else "Unknown",
                    "memory_allocated": torch.cuda.memory_allocated(0) if torch.cuda.device_count() > 0 else 0,
                    "memory_reserved": torch.cuda.memory_reserved(0) if torch.cuda.device_count() > 0 else 0
                }
            except Exception as e:
                gpu_info = {"error": f"Failed to get GPU info: {str(e)}"}
        
        return {
            "status": "healthy", 
            "device": device_info,
            "cuda_available": cuda_available,
            "gpu_info": gpu_info,
            "pytorch_version": torch.__version__,
            "models_loaded": {
                "face": model_manager.face_app is not None,
                "blip": model_manager.blip_model is not None,
                "clip": model_manager.clip_model is not None
            }
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}