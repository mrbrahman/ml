# AI Photo Analysis Service

A Python-based AI/ML service for comprehensive photo analysis including face recognition, clustering, and image description generation.

## Features

- **Face Recognition**: Detect, cluster, and identify faces using InsightFace
- **Image Descriptions**: Generate detailed captions using BLIP-2
- **Vector Storage**: FAISS-based storage for face embeddings and clustering
- **GPU Support**: Automatic GPU detection and utilization
- **REST API**: Clean interface for Node.js integration

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download all AI models (run this on a network with access to GitHub/Hugging Face):
```bash
python download_all_models.py
```

   **Note**: If you're on a restricted network, the service will use fallback mechanisms for missing models. Face detection will still work with just the InsightFace model.

## Usage

1. Start the service:
```bash
python main.py
```

2. The API will be available at `http://localhost:8000`

## API Endpoints

### POST /analyze
Analyze an image for faces and generate description.

**Request:**
```json
{
  "image_id": "uuid-from-nodejs",
  "image_path": "/path/to/image.jpg"
}
```

**Response:**
```json
{
  "image_id": "uuid-from-nodejs",
  "image_path": "/path/to/image.jpg",
  "faces": [
    {
      "bbox": [x, y, w, h],
      "confidence": 0.95,
      "cluster_id": "cluster_abc123",
      "person_name": null,
      "gender": "M",
      "age": 25,
      "landmarks": {
        "left_eye": [x1, y1],
        "right_eye": [x2, y2],
        "nose": [x3, y3],
        "left_mouth": [x4, y4],
        "right_mouth": [x5, y5]
      },
      "pose": {
        "yaw": -5.2,
        "pitch": 2.1,
        "roll": 1.8
      }
    }
  ],
  "description": "A detailed description of the image",
  "models_used": {
    "face_detection": "buffalo_l",
    "image_captioning": "blip2-opt-2.7b"
  }
}
```

### POST /faces/recognize
Face recognition only - detect and identify faces without image description.

**Request:**
```json
{
  "image_id": "uuid-from-nodejs",
  "image_path": "/path/to/image.jpg"
}
```

**Response:**
```json
{
  "image_id": "uuid-from-nodejs",
  "image_path": "/path/to/image.jpg",
  "faces": [
    {
      "bbox": [x, y, w, h],
      "confidence": 0.95,
      "cluster_id": "cluster_abc123",
      "person_name": "John Doe",
      "gender": "M",
      "age": 25,
      "landmarks": {
        "left_eye": [x1, y1],
        "right_eye": [x2, y2],
        "nose": [x3, y3],
        "left_mouth": [x4, y4],
        "right_mouth": [x5, y5]
      },
      "pose": {
        "yaw": -5.2,
        "pitch": 2.1,
        "roll": 1.8
      }
    }
  ],
  "models_used": {
    "face_detection": "buffalo_l"
  }
}
```

### PUT /faces/{cluster_id}
Assign a name to a face cluster.

**Request:**
```json
{
  "name": "John Doe"
}
```

### POST /search/text
Search for images using text queries.

**Request:**
```json
{
  "query": "person smiling outdoors",
  "limit": 10
}
```

**Response:**
```json
{
  "query": "person smiling outdoors",
  "results": [
    {
      "image_id": "uuid-1",
      "score": 0.85
    }
  ]
}
```

### POST /search/similar
Find visually similar images.

**Request:**
```json
{
  "image_id": "reference-uuid",
  "image_path": "/path/to/reference/image.jpg"
}
```

### POST /train
Train face recognition from a dataset of known faces.

**Directory Structure Training:**
```json
{
  "dataset_path": "/path/to/training_data",
  "dataset_type": "directory"
}
```

Expected directory structure:
```
training_data/
├── john_doe/
│   ├── photo1.jpg
│   └── photo2.jpg
└── jane_smith/
    ├── photo1.jpg
    └── photo2.jpg
```

**JSON Training:**
```json
{
  "dataset_path": "/path/to/training.json",
  "dataset_type": "json"
}
```

JSON format:
```json
{
  "John Doe": ["/path/to/john1.jpg", "/path/to/john2.jpg"],
  "Jane Smith": ["/path/to/jane1.jpg"]
}
```

### GET /faceinfo
Get information about face clusters and recognition statistics.

**Response:**
```json
{
  "total_clusters": 15,
  "named_clusters": 8,
  "clusters": [
    {
      "cluster_id": "cluster_abc123",
      "name": "John Doe",
      "face_count": 12
    },
    {
      "cluster_id": "cluster_def456",
      "name": null,
      "face_count": 3
    }
  ]
}
```

### GET /health
Health check endpoint with detailed system information.

**Response:**
```json
{
  "status": "healthy",
  "device": "cuda",
  "cuda_available": true,
  "gpu_info": {
    "gpu_count": 1,
    "current_device": 0,
    "device_name": "NVIDIA GeForce RTX 4090",
    "memory_allocated": 2048576,
    "memory_reserved": 4194304
  },
  "pytorch_version": "2.0.1",
  "models_loaded": {
    "face": true,
    "blip": true,
    "clip": true
  }
}
```

## Project Structure

```
project/
├── src/                    # Application source code
│   ├── api/               # FastAPI routes and handlers
│   ├── services/          # Business logic layer
│   ├── data/              # Data access and vector operations
│   ├── core/              # ML model operations
│   ├── infrastructure/    # Configuration and model management
│   └── schemas/           # API request/response models
├── data/                  # Runtime data storage
│   ├── faiss_indices/     # FAISS vector indices
│   └── training_data/     # Training datasets
├── main.py                # Application entry point
├── download_all_models.py # Model download utility
└── requirements.txt       # Python dependencies
```

## Configuration

Edit `src/infrastructure/config.py` to modify:
- Model settings
- Similarity thresholds
- FAISS storage paths
- API configuration

## GPU Support

The service automatically detects and uses GPU if available. Models will be loaded on CUDA device for faster inference.

## Training Dataset

Train the system with known faces before use:

**Option 1: Directory Structure**
```bash
curl -X POST "http://localhost:8000/train" \
  -H "Content-Type: application/json" \
  -d '{"dataset_path": "/path/to/training_data", "dataset_type": "directory"}'
```

**Option 2: JSON File**
```bash
curl -X POST "http://localhost:8000/train" \
  -H "Content-Type: application/json" \
  -d '{"dataset_path": "/path/to/training.json", "dataset_type": "json"}'
```

## Storage

- Face embeddings and clusters stored in FAISS indices
- No metadata storage - all returned to calling application
- Persistent storage in `data/faiss_indices/` directory