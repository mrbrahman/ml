import uvicorn
import os
import sys
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import *
from app import services
from app.config import HOST, PORT, LOG_LEVEL, DEVICE
from app.utils.logging import setup_logging, get_logger

# Setup logging
setup_logging(LOG_LEVEL)
logger = get_logger(__name__)

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
    logger.info(f"Starting image analysis for {request.image_id} at {request.image_path}")
    
    if not os.path.exists(request.image_path):
        logger.error(f"Image file not found: {request.image_path}")
        raise HTTPException(status_code=404, detail="Image file not found")
    
    try:
        result = services.images.analyze_composite(request.image_id, request.image_path, request.orientation, request.xmp_regions, request.save_annotated)
        logger.info(f"Image analysis completed for {request.image_id}")
        return result
    except Exception as e:
        logger.error(f"Analysis failed for {request.image_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/faces/recognize", response_model=FaceRecognitionResponse)
async def recognize_faces_endpoint(request: AnalyzeImageRequest):
    logger.info(f"Starting face recognition for {request.image_id}")
    
    if not os.path.exists(request.image_path):
        logger.error(f"Image file not found: {request.image_path}")
        raise HTTPException(status_code=404, detail="Image file not found")
    
    try:
        result = services.faces.recognize(request.image_id, request.image_path, request.orientation, request.save_annotated, request.xmp_regions)
        logger.info(f"Face recognition completed for {request.image_id}, found {len(result.faces)} faces")
        return result
    except Exception as e:
        logger.error(f"Face recognition failed for {request.image_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Face recognition failed: {str(e)}")

@app.post("/images/caption", response_model=ImageCaptionResponse)
async def caption_image_endpoint(request: ImageCaptionRequest):
    logger.info(f"Starting image captioning for {request.image_id}")
    
    if not os.path.exists(request.image_path):
        logger.error(f"Image file not found: {request.image_path}")
        raise HTTPException(status_code=404, detail="Image file not found")
    
    try:
        result = services.images.generate_caption(request.image_id, request.image_path)
        logger.info(f"Image captioning completed for {request.image_id}")
        return result
    except Exception as e:
        logger.error(f"Image captioning failed for {request.image_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Image captioning failed: {str(e)}")

@app.post("/images/encode", response_model=ImageEncodeResponse)
async def encode_image_endpoint(request: ImageEncodeRequest):
    if not os.path.exists(request.image_path):
        raise HTTPException(status_code=404, detail="Image file not found")
    
    try:
        return services.images.encode_image(request.image_id, request.image_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image encoding failed: {str(e)}")

@app.put("/faces/{cluster_id}", response_model=NameClusterResponse)
async def name_face_cluster_endpoint(cluster_id: str, request: NameClusterRequest):
    logger.info(f"Naming cluster {cluster_id} as '{request.name}'")
    success = services.storage.name_face_cluster(cluster_id, request.name)
    
    if success:
        logger.info(f"Successfully named cluster {cluster_id} as '{request.name}'")
        return NameClusterResponse(success=True, message=f"Cluster {cluster_id} named as '{request.name}'")
    else:
        logger.warning(f"Cluster {cluster_id} not found")
        return NameClusterResponse(success=False, message=f"Cluster {cluster_id} not found")



@app.post("/search/text", response_model=SearchResponse)
async def search_by_text_endpoint(request: SearchRequest):
    logger.info(f"Text search query: '{request.query}', limit: {request.limit}")
    try:
        search_results = services.search.by_text(request.query, request.limit)
        logger.info(f"Text search completed, found {len(search_results)} results")
        return SearchResponse(query=request.query, results=search_results)
    except Exception as e:
        logger.error(f"Text search failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.post("/search/similar", response_model=SearchResponse)
async def search_similar_images_endpoint(request: AnalyzeImageRequest):
    if not os.path.exists(request.image_path):
        raise HTTPException(status_code=404, detail="Image file not found")
    
    try:
        search_results = services.search.find_similar(request.image_path, limit=10)
        return SearchResponse(query=f"Similar to {request.image_path}", results=search_results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.post("/faces/update-name", response_model=NameClusterResponse)
async def update_face_cluster_name_endpoint(request: UpdatePersonNameRequest):
    logger.info(f"Updating person name from '{request.old_name}' to '{request.new_name}'")
    updated_clusters = 0
    updated_faces = 0
    face_clusters, face_cluster_names = services.storage.get_face_clusters()
    for cluster_id, name in list(face_cluster_names.items()):
        if name == request.old_name:
            services.storage.name_face_cluster(cluster_id, request.new_name)
            updated_clusters += 1
            updated_faces += len(face_clusters.get(cluster_id, []))
    
    if updated_clusters > 0:
        logger.info(f"Updated {updated_clusters} clusters and {updated_faces} faces from '{request.old_name}' to '{request.new_name}'")
        return NameClusterResponse(
            success=True,
            message=f"Updated '{request.old_name}' to '{request.new_name}' ({updated_clusters} clusters, {updated_faces} faces)"
        )
    else:
        logger.warning(f"Person '{request.old_name}' not found")
        return NameClusterResponse(success=False, message=f"Person '{request.old_name}' not found")

@app.post("/faces/correct", response_model=CorrectFaceAssignmentResponse)
async def correct_face_assignment_endpoint(request: CorrectFaceAssignmentRequest):
    logger.info(f"Correcting face assignment for {request.image_id} to '{request.person_name}'")
    try:
        success, cluster_id, action, message = services.clustering.correct_face_assignment(request.image_id, request.person_name)
        logger.info(f"Face correction completed: {action} for {request.image_id}")
        
        return CorrectFaceAssignmentResponse(
            success=success,
            message=message,
            cluster_id=cluster_id,
            action_taken=action
        )
    except Exception as e:
        logger.error(f"Face correction failed for {request.image_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Face correction failed: {str(e)}")

@app.get("/faceinfo", response_model=InfoResponse)
async def get_face_info(cluster_id: str = None, person_name: str = None):
    return services.faces.get_cluster_info(cluster_id, person_name)

@app.get("/faces/suggestions")
async def get_cluster_name_suggestions(cluster_id: str = None, min_similarity: float = None):
    try:
        suggestions = services.clustering.get_cluster_name_suggestions(cluster_id, min_similarity)
        return {"suggestions": suggestions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get suggestions: {str(e)}")

@app.post("/images/annotate", response_model=AnnotateImageResponse)
async def annotate_image_endpoint(request: AnnotateImageRequest):
    if not os.path.exists(request.image_path):
        raise HTTPException(status_code=404, detail="Image file not found")
    
    try:
        annotated_path = services.create_enriched_image(
            request.image_path, 
            request.faces, 
            request.output_dir, 
            request.unmatched_input_faces
        )
        return AnnotateImageResponse(
            success=True,
            annotated_image_path=annotated_path,
            message=f"Annotated image saved to {annotated_path}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Annotation failed: {str(e)}")

@app.get("/health")
async def health_check():
    try:
        cuda_available = torch.cuda.is_available()
        
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
        
        model_status = services.model_loader.get_model_status()
        
        return {
            "status": "healthy", 
            "device": DEVICE,
            "device_mode": os.getenv("DEVICE_MODE", "auto").lower(),
            "cuda_available": cuda_available,
            "gpu_info": gpu_info,
            "pytorch_version": torch.__version__,
            "model_status": model_status
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    logger.info(f"Starting AI Photo Analysis Service on {HOST}:{PORT}")
    
    uvicorn.run(app, host=HOST, port=PORT, log_config=None)
