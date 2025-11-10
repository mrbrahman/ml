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
  "image_path": "/path/to/image.jpg",
  "save_annotated": false,
  "xmp_faces": [
    {
      "name": "John Doe",
      "x": 0.45,
      "y": 0.25,
      "w": 0.12,
      "h": 0.18
    }
  ],
  "xmp_regions": {
    "AppliedToDimensions": {"H": 2160, "Unit": "pixel", "W": 2880},
    "RegionList": [
      {
        "Area": {"H": 0.0791667, "Unit": "normalized", "W": 0.0496528, "X": 0.673438, "Y": 0.477083},
        "Name": "John Doe",
        "Type": "Face"
      }
    ]
  }
}
```

- `save_annotated` (optional): When `true`, saves an annotated copy of the image with bounding boxes and face labels to `data/annotated_images/`
- `xmp_faces` (optional): Array of face regions from EXIF/XMP metadata for automatic face labeling
- `xmp_regions` (optional): Raw XMP regions object from exiftool-vendored for automatic conversion and face labeling

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
      },
      "xmp_matched": true,
      "xmp_match_confidence": 0.87
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
  "image_path": "/path/to/image.jpg",
  "save_annotated": false,
  "xmp_faces": [
    {
      "name": "John Doe",
      "x": 0.45,
      "y": 0.25,
      "w": 0.12,
      "h": 0.18
    }
  ],
  "xmp_regions": {
    "AppliedToDimensions": {"H": 2160, "Unit": "pixel", "W": 2880},
    "RegionList": [
      {
        "Area": {"H": 0.0791667, "Unit": "normalized", "W": 0.0496528, "X": 0.673438, "Y": 0.477083},
        "Name": "John Doe",
        "Type": "Face"
      }
    ]
  }
}
```

- `save_annotated` (optional): When `true`, saves an annotated copy of the image with bounding boxes and face labels to `data/annotated_images/`
- `xmp_faces` (optional): Array of face regions from EXIF/XMP metadata for automatic face labeling
- `xmp_regions` (optional): Raw XMP regions object from exiftool-vendored for automatic conversion and face labeling

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
      },
      "xmp_matched": true,
      "xmp_match_confidence": 0.87
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

### POST /faces/update-name
Update the name of an existing face cluster.

**Request:**
```json
{
  "old_name": "John Doe",
  "new_name": "Jane Smith"
}
```

**Response:**
```json
{
  "success": true,
  "message": "Updated 'John Doe' to 'Jane Smith' (2 clusters, 15 faces)"
}
```

### POST /faces/correct
Correct a face assignment by providing the correct person name. The system will automatically move the face to the best matching cluster for that person or create a new one.

**Request:**
```json
{
  "image_id": "uuid-from-nodejs",
  "person_name": "John Doe"
}
```

**Response:**
```json
{
  "success": true,
  "message": "Moved to existing cluster for John Doe",
  "cluster_id": "cluster_def456",
  "action_taken": "moved_to_existing"
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
Train face recognition from a dataset of known faces. Supports two dataset formats:

**Method 1: Directory Structure**
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

**Method 2: JSON File**
```json
{
  "dataset_path": "/path/to/training.json",
  "dataset_type": "json"
}
```

The JSON file should contain person names as keys and arrays of image paths as values:
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
│   ├── annotated_images/  # Annotated images with face boxes
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

Train the system with known faces before use. Two methods are supported:

**Method 1: Directory Structure**
```bash
curl -X POST "http://localhost:8000/train" \
  -H "Content-Type: application/json" \
  -d '{"dataset_path": "/path/to/training_data", "dataset_type": "directory"}'
```

**Method 2: JSON File**
```bash
curl -X POST "http://localhost:8000/train" \
  -H "Content-Type: application/json" \
  -d '{"dataset_path": "/path/to/training.json", "dataset_type": "json"}'
```

## Storage

- Face embeddings and clusters stored in FAISS indices
- No metadata storage - all returned to calling application
- Persistent storage in `data/faiss_indices/` directory