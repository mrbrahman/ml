import uvicorn
import os
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import *
from app.services import analyze_image, recognize_faces, caption_image, encode_image, search_by_text, find_similar_images, get_cluster_info
from app.core.image_analysis import search
from app.core import model_loader
from app.core.face_recognition import storage as face_storage
from app.core.face_recognition import recognition as face_recognition
from app.config import HOST, PORT

app = FastAPI(title="AI Photo Analysis Service", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/analyze", response_model=CompositeAnalyzeResponse)
async def analyze_image_endpoint(request: AnalyzeImageRequest):
    if not os.path.exists(request.image_path):
        raise HTTPException(status_code=404, detail="Image file not found")
    
    try:
        return analyze_image(request.image_id, request.image_path, request.xmp_faces, request.xmp_regions, request.save_annotated)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/faces/recognize", response_model=FaceRecognitionResponse)
async def recognize_faces_endpoint(request: AnalyzeImageRequest):
    if not os.path.exists(request.image_path):
        raise HTTPException(status_code=404, detail="Image file not found")
    
    try:
        return recognize_faces(request.image_id, request.image_path, request.save_annotated, request.xmp_faces, request.xmp_regions)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Face recognition failed: {str(e)}")

@app.post("/images/caption", response_model=ImageCaptionResponse)
async def caption_image_endpoint(request: ImageCaptionRequest):
    if not os.path.exists(request.image_path):
        raise HTTPException(status_code=404, detail="Image file not found")
    
    try:
        return caption_image(request.image_id, request.image_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image captioning failed: {str(e)}")

@app.post("/images/encode", response_model=ImageEncodeResponse)
async def encode_image_endpoint(request: ImageEncodeRequest):
    if not os.path.exists(request.image_path):
        raise HTTPException(status_code=404, detail="Image file not found")
    
    try:
        return encode_image(request.image_id, request.image_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image encoding failed: {str(e)}")

@app.put("/faces/{cluster_id}", response_model=NameClusterResponse)
async def name_face_cluster_endpoint(cluster_id: str, request: NameClusterRequest):
    success = face_storage.name_face_cluster(cluster_id, request.name)
    
    if success:
        return NameClusterResponse(success=True, message=f"Cluster {cluster_id} named as '{request.name}'")
    else:
        return NameClusterResponse(success=False, message=f"Cluster {cluster_id} not found")



@app.post("/search/text", response_model=SearchResponse)
async def search_by_text_endpoint(request: SearchRequest):
    try:
        search_results = search_by_text(request.query, request.limit)
        return SearchResponse(query=request.query, results=search_results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.post("/search/similar", response_model=SearchResponse)
async def search_similar_images_endpoint(request: AnalyzeImageRequest):
    if not os.path.exists(request.image_path):
        raise HTTPException(status_code=404, detail="Image file not found")
    
    try:
        search_results = find_similar_images(request.image_path, limit=10)
        return SearchResponse(query=f"Similar to {request.image_path}", results=search_results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.post("/faces/update-name", response_model=NameClusterResponse)
async def update_face_cluster_name_endpoint(request: UpdatePersonNameRequest):
    updated_clusters = 0
    updated_faces = 0
    face_clusters, face_cluster_names = face_storage.get_face_clusters()
    for cluster_id, name in list(face_cluster_names.items()):
        if name == request.old_name:
            face_storage.name_face_cluster(cluster_id, request.new_name)
            updated_clusters += 1
            updated_faces += len(face_clusters.get(cluster_id, []))
    
    if updated_clusters > 0:
        return NameClusterResponse(
            success=True,
            message=f"Updated '{request.old_name}' to '{request.new_name}' ({updated_clusters} clusters, {updated_faces} faces)"
        )
    else:
        return NameClusterResponse(success=False, message=f"Person '{request.old_name}' not found")

@app.post("/faces/correct", response_model=CorrectFaceAssignmentResponse)
async def correct_face_assignment_endpoint(request: CorrectFaceAssignmentRequest):
    try:
        success, cluster_id, action, message = face_recognition.correct_face_assignment(request.image_id, request.person_name)
        
        return CorrectFaceAssignmentResponse(
            success=success,
            message=message,
            cluster_id=cluster_id,
            action_taken=action
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Face correction failed: {str(e)}")

@app.get("/faceinfo", response_model=InfoResponse)
async def get_face_info():
    return get_cluster_info()

@app.get("/health")
async def health_check():
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
        
        model_status = model_loader.get_model_status()
        
        return {
            "status": "healthy", 
            "device": device_info,
            "cuda_available": cuda_available,
            "gpu_info": gpu_info,
            "pytorch_version": torch.__version__,
            "model_status": model_status
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
