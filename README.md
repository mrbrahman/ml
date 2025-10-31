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
      "person_name": null
    }
  ],
  "description": "A detailed description of the image"
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
      "person_name": "John Doe"
    }
  ]
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

### GET /health
Health check endpoint.

## Configuration

Edit `config.py` to modify:
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
- Persistent storage in `faiss_indices/` directory