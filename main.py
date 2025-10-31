from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import torch
from schemas import AnalyzeImageRequest, AnalyzeImageResponse, FaceInfo, NameClusterRequest, NameClusterResponse, SearchRequest, SearchResponse, SearchResult, TrainRequest, TrainResponse, FaceRecognitionResponse
from models import ModelManager
from vector_store import VectorStore
from face_trainer import FaceTrainer
from config import HOST, PORT, MODEL_NAMES

# Global instances - loaded at startup
model_manager = ModelManager()
vector_store = VectorStore()
face_trainer = FaceTrainer()

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
async def analyze_image(request: AnalyzeImageRequest):
    """Analyze image for faces and generate description"""
    
    # Validate image path exists
    if not os.path.exists(request.image_path):
        raise HTTPException(status_code=404, detail="Image file not found")
    
    try:
        # Remove existing embeddings for this image first
        vector_store.remove_image_embeddings(request.image_id)
        
        # Always process image fully (allows for model updates)
        faces_data = model_manager.detect_faces(request.image_path)
        
        # Process each face with fresh embeddings
        faces_info = []
        for face_data in faces_data:
            cluster_id, person_name = vector_store.add_face_embedding(
                request.image_id, 
                face_data['embedding']
            )
            
            faces_info.append(FaceInfo(
                bbox=face_data['bbox'],
                confidence=face_data['confidence'],
                cluster_id=cluster_id,
                person_name=person_name,
                gender=face_data.get('gender'),
                age=face_data.get('age'),
                landmarks=face_data.get('landmarks'),
                pose=face_data.get('pose')
                # landmarks_3d=face_data.get('landmarks_3d'),
                # landmarks_2d_106=face_data.get('landmarks_2d_106')
            ))
        
        # Generate image description
        description = model_manager.generate_description(request.image_path)
        
        # Replace visual and text embeddings (removes old ones first)
        clip_embedding = model_manager.get_clip_embedding(request.image_path)
        vector_store.replace_visual_embedding(request.image_id, clip_embedding)
        vector_store.replace_text_embedding(request.image_id, clip_embedding)
        
        return AnalyzeImageResponse(
            image_id=request.image_id,
            image_path=request.image_path,
            faces=faces_info,
            description=description,
            models_used={
                "face_detection": MODEL_NAMES["face_detection"],
                "image_description": MODEL_NAMES["image_description"],
                "embeddings": MODEL_NAMES["visual_similarity"]
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/faces/recognize", response_model=FaceRecognitionResponse)
async def recognize_faces(request: AnalyzeImageRequest):
    """Face recognition only - detect and identify faces without image description"""
    
    if not os.path.exists(request.image_path):
        raise HTTPException(status_code=404, detail="Image file not found")
    
    try:
        # Remove existing embeddings for this image first
        vector_store.remove_image_embeddings(request.image_id)
        
        # Always process image fully (allows for model updates)
        faces_data = model_manager.detect_faces(request.image_path)
        
        # Process each face with fresh embeddings
        faces_info = []
        for face_data in faces_data:
            cluster_id, person_name = vector_store.add_face_embedding(
                request.image_id, 
                face_data['embedding']
            )
            
            faces_info.append(FaceInfo(
                bbox=face_data['bbox'],
                confidence=face_data['confidence'],
                cluster_id=cluster_id,
                person_name=person_name,
                gender=face_data.get('gender'),
                age=face_data.get('age'),
                landmarks=face_data.get('landmarks'),
                pose=face_data.get('pose')
                # landmarks_3d=face_data.get('landmarks_3d'),
                # landmarks_2d_106=face_data.get('landmarks_2d_106')
            ))
        
        return FaceRecognitionResponse(
            image_id=request.image_id,
            image_path=request.image_path,
            faces=faces_info,
            models_used={
                "face_detection": MODEL_NAMES["face_detection"]
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Face recognition failed: {str(e)}")

@app.put("/faces/{cluster_id}", response_model=NameClusterResponse)
async def name_face_cluster(cluster_id: str, request: NameClusterRequest):
    """Assign name to a face cluster"""
    
    success = vector_store.name_face_cluster(cluster_id, request.name)
    
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
async def train_faces(request: TrainRequest):
    """Train face recognition from dataset"""
    try:
        if request.dataset_type == "directory":
            success = face_trainer.train_from_dataset(request.dataset_path)
        elif request.dataset_type == "json":
            success = face_trainer.train_from_json(request.dataset_path)
        else:
            raise HTTPException(status_code=400, detail="dataset_type must be 'directory' or 'json'")
        
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
async def search_by_text(request: SearchRequest):
    """Search for images using text query"""
    try:
        # Get text embedding
        text_embedding = model_manager.get_text_embedding(request.query)
        
        # Search for similar images
        results = vector_store.search_by_text(text_embedding, k=request.limit)
        
        search_results = [
            SearchResult(image_id=image_id, score=score)
            for image_id, score in results
        ]
        
        return SearchResponse(
            query=request.query,
            results=search_results
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.post("/search/similar", response_model=SearchResponse)
async def search_similar_images(request: AnalyzeImageRequest):
    """Find visually similar images"""
    if not os.path.exists(request.image_path):
        raise HTTPException(status_code=404, detail="Image file not found")
    
    try:
        # Get image embedding
        clip_embedding = model_manager.get_clip_embedding(request.image_path)
        
        # Search for similar images
        results = vector_store.search_similar_images(clip_embedding, k=10)
        
        search_results = [
            SearchResult(image_id=image_id, score=score)
            for image_id, score in results
        ]
        
        return SearchResponse(
            query=f"Similar to {request.image_path}",
            results=search_results
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

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
                "face": model_manager.face_app is not None if model_manager else False,
                "blip": model_manager.blip_model is not None if model_manager else False,
                "clip": model_manager.clip_model is not None if model_manager else False
            }
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)