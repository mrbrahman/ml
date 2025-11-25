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
python server.py
```

2. The API will be available at `http://localhost:8000`

## API Endpoints

### Face Recognition

#### POST /faces/recognize
Face recognition only - detect and identify faces without image description.

**Request:**
```json
{
  "image_id": "uuid-from-nodejs",
  "image_path": "/path/to/image.jpg",
  "save_annotated": false,
  "orientation": 1,
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
- `orientation` (required): EXIF orientation value (1-8) for coordinate transformation
- `xmp_regions` (optional): Raw XMP regions object from exiftool-vendored for automatic face labeling

**Response:**
```json
{
  "image_id": "uuid-from-nodejs",
  "image_path": "/path/to/image.jpg",
  "faces": [
    {
      "bbox": [x, y, w, h],
      "confidence": 0.95,
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
      "cluster": {
        "cluster_id": "cluster_abc123",
        "name": "John Doe",
        "confidence": 0.85,
        "consensus_count": 3,
        "reference_image_ids": ["img1", "img2"],
        "is_new_cluster": false,
        "centroid": [0.5, 0.4]
      },
      "input_face_match": {
        "matched": true,
        "name": "John Doe",
        "confidence": 0.87,
        "match_strategy": "centroid_distance",
        "input_bbox": [100, 80, 200, 180],
        "centroid": [0.5, 0.4]
      },
      "name_mismatch": false
    }
  ],
  "unmatched_input_faces": [
    {
      "name": "Jane Smith",
      "x": 0.25,
      "y": 0.35,
      "w": 0.10,
      "h": 0.15,
      "centroid": [0.3, 0.425]
    }
  ],
  "models_used": {
    "face_detection": "buffalo_l"
  }
}
```

#### PUT /faces/{cluster_id}
Assign a name to a face cluster.

**Request:**
```json
{
  "name": "John Doe"
}
```

#### POST /faces/update-name
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

#### POST /faces/correct
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

#### GET /faceinfo
Get information about face clusters and recognition statistics.

**Query Parameters:**
- `cluster_id` (optional): Filter results to specific cluster ID
- `person_name` (optional): Filter results to clusters with specific person name

**Usage Examples:**
- `GET /faceinfo` - Returns all clusters
- `GET /faceinfo?cluster_id=cluster_abc123` - Returns only the specified cluster
- `GET /faceinfo?person_name=John%20Doe` - Returns all clusters named "John Doe"

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

#### GET /faces/suggestions
Get name suggestions for unnamed clusters based on centroid analysis with nearest named clusters.

**Query Parameters:**
- `cluster_id` (optional): Specific cluster ID to get suggestions for
- `min_similarity` (optional): Minimum similarity score threshold (default: 0.6)

**Response:**
```json
{
  "suggestions": [
    {
      "cluster_id": "cluster_def456",
      "face_count": 3,
      "suggested_name": "John Doe",
      "similarity_score": 0.82,
      "reference_cluster_id": "cluster_abc123"
    }
  ]
}
```

### Image Analysis

#### POST /images/caption
Generate image description using BLIP-2.

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
  "description": "A detailed description of the image",
  "models_used": {
    "image_captioning": "blip2-opt-2.7b"
  }
}
```

#### POST /images/encode
Generate and store image embeddings using CLIP for search functionality.

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
  "embedding_stored": true,
  "models_used": {
    "image_encoding": "clip-vit-base-patch32"
  }
}
```

### Combined Analysis

#### POST /analyze
Analyze an image for faces and generate description. This endpoint combines the functionality of `/faces/recognize`, `/images/caption`, and `/images/encode`.

**Request:**
```json
{
  "image_id": "uuid-from-nodejs",
  "image_path": "/path/to/image.jpg",
  "save_annotated": false,
  "orientation": 1,
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
- `orientation` (required): EXIF orientation value (1-8) for coordinate transformation
- `xmp_regions` (optional): Raw XMP regions object from exiftool-vendored for automatic face labeling

**Response:**
```json
{
  "face_recognition": {
    "image_id": "uuid-from-nodejs",
    "image_path": "/path/to/image.jpg",
    "faces": [
      {
        "bbox": [x, y, w, h],
        "confidence": 0.95,
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
        "cluster": {
          "cluster_id": "cluster_abc123",
          "name": null,
          "confidence": 0.85,
          "consensus_count": 3,
          "reference_image_ids": ["img1", "img2"],
          "is_new_cluster": false,
          "centroid": [0.5, 0.4]
        },
        "input_face_match": {
          "matched": true,
          "name": "John Doe",
          "confidence": 0.87,
          "match_strategy": "centroid_distance",
          "input_bbox": [100, 80, 200, 180],
          "centroid": [0.5, 0.4]
        },
        "name_mismatch": true
      }
    ],
    "unmatched_input_faces": [
      {
        "name": "Jane Smith",
        "x": 0.25,
        "y": 0.35,
        "w": 0.10,
        "h": 0.15,
        "centroid": [0.3, 0.425]
      }
    ],
    "models_used": {
      "face_detection": "buffalo_l"
    }
  },
  "image_caption": {
    "image_id": "uuid-from-nodejs",
    "image_path": "/path/to/image.jpg",
    "description": "A detailed description of the image",
    "models_used": {
      "image_captioning": "blip2-opt-2.7b"
    }
  },
  "image_encode": {
    "image_id": "uuid-from-nodejs",
    "image_path": "/path/to/image.jpg",
    "embedding_stored": true,
    "models_used": {
      "image_encoding": "clip-vit-base-patch32"
    }
  }
}
```

### Search

#### POST /search/text
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

#### POST /search/similar
Find visually similar images.

**Request:**
```json
{
  "image_id": "reference-uuid",
  "image_path": "/path/to/reference/image.jpg"
}
```

### System

#### GET /health
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
├── server.py              # FastAPI app + routes + uvicorn.run()
├── download_all_models.py # Model download utility
├── app/                   # Main application package
│   ├── config.py          # Configuration
│   ├── schemas.py         # Pydantic models
│   ├── services.py        # Business logic
│   ├── core/              # Core ML functionality
│   │   ├── model_loader.py        # Shared model loading and caching
│   │   ├── face_recognition/      # Face recognition module
│   │   │   ├── detection.py       # Face detection
│   │   │   ├── clustering.py      # Face clustering algorithms
│   │   │   ├── manager.py         # Face recognition manager
│   │   │   ├── storage.py         # FAISS storage operations
│   │   │   ├── xmp_processor.py   # XMP metadata processing
│   │   │   └── annotator.py       # Image annotation
│   │   └── image_analysis/        # Image description & search module
│   │       ├── captioning.py      # Image description generation
│   │       ├── embeddings.py      # Image embedding generation
│   │       ├── manager.py         # Image analysis manager
│   │       ├── search.py          # Vector search functionality
│   │       └── storage.py         # Vector storage operations
│   └── utils/             # Utility functions
├── data/                  # Runtime data storage
│   ├── faiss_indices/     # FAISS vector indices
│   ├── annotated_images/  # Annotated images with face boxes
│   └── training_data/     # Training datasets
└── requirements.txt       # Python dependencies
```

## Configuration

Edit `app/config.py` to modify:
- Model settings
- Similarity thresholds
- FAISS storage paths
- API configuration
- Logging settings

### Environment Variables

- `LOG_LEVEL`: Set logging level (DEBUG, INFO, WARNING, ERROR). Default: INFO
- `LOG_FILE`: Path to log file. Default: console output only

```bash
export LOG_LEVEL=DEBUG
export LOG_FILE=/path/to/custom/logfile.log  # Enable file logging
```

**Logging Behavior:**
- **Interactive mode (TTY)**: Colored output with timestamps
- **Non-interactive mode**: Plain format without timestamps (for journalctl)
- **File logging**: Always includes full timestamps when LOG_FILE is set

## GPU Support

The service automatically detects and uses GPU if available. Models will be loaded on CUDA device for faster inference.

## Training Dataset

Training is now integrated into the main analysis endpoints rather than using a separate `/train` endpoint. While previous versions supported traditional training with cropped face thumbnails in directory structures like:

```
training_data/
├── john_doe/
│   ├── photo1.jpg
│   └── photo2.jpg
└── jane_smith/
    ├── photo1.jpg
    └── photo2.jpg
```

... this approach had limitations as InsightFace often failed to detect faces in small thumbnail images, while successfully detecting the same faces in full-resolution photos.

**Current Training Approach:**
Training now occurs automatically through the `/analyze` and `/faces/recognize` endpoints when `xmp_regions` metadata is provided. This allows the system to learn from full-resolution images with labeled face regions.

**Processing Order:**
1. First, process images with labeled faces (using `xmp_regions`)
2. Then, process unlabeled images for automatic face clustering and recognition

This approach leverages the superior face detection capabilities on full images while maintaining accurate face labeling through metadata.

## Storage

- Face embeddings and clusters stored in FAISS indices
- No metadata storage - all returned to calling application
- Persistent storage in `data/faiss_indices/` directory
- Application logs output to console by default (file logging optional)

## Face Recognition Response Structure

The face recognition response uses a grouped structure for better organization:

### Face Object Fields

- **Basic Detection**: `bbox`, `confidence`, `gender`, `age`, `landmarks`, `pose`
- **Final Identity**: `person_name` (resolved from XMP or cluster matching)
- **Cluster Information**: `cluster` object containing:
  - `cluster_id`: Unique cluster identifier
  - `name`: Name assigned to cluster (may be null)
  - `confidence`: Similarity score for cluster match (0.0-1.0)
  - `consensus_count`: Number of faces that agreed on this cluster
  - `reference_image_ids`: Image IDs of faces used for matching
  - `is_new_cluster`: True if this face created a new cluster
  - `centroid`: Normalized centroid coordinates [x, y]
- **Input Face Match**: `input_face_match` object containing:
  - `matched`: True if face matched XMP region data
  - `name`: Name from XMP metadata (may be null)
  - `confidence`: Distance-based confidence for centroid matching
  - `match_strategy`: Matching method used ("centroid_distance")
  - `input_bbox`: Original XMP region coordinates in pixels
  - `centroid`: Normalized centroid coordinates [x, y]
- **Name Validation**: `name_mismatch` boolean indicating if cluster name differs from XMP name

### Unmatched Input Faces

The `unmatched_input_faces` array contains faces from the input (`xmp_regions`) that were not detected in the image. This helps identify:
- Faces that may be too small, blurry, or obscured for detection
- Incorrectly tagged face regions in metadata
- Faces that require manual review or re-tagging

Each unmatched face includes the original normalized coordinates, centroid, and name from the input.

### Centroid-Based Matching

The system uses centroid-based least distance matching for XMP face regions:
- **Centroid Calculation**: Each face (detected and XMP) has a normalized centroid [x, y]
- **Distance Matching**: For each XMP face, finds the closest detected face by Euclidean distance
- **Confidence**: Distance is converted to confidence (1.0 - distance)
- **Threshold**: Maximum distance of 0.1 (normalized coordinates) for valid matches

### Name Mismatch Detection

The system compares names from two sources:
- **Cluster matching**: Face similarity-based clustering assigns faces to existing named clusters
- **XMP metadata**: Photo metadata contains manually tagged face regions with names

When these disagree, `name_mismatch` is set to `true`, indicating potential mislabeling that may need manual review.

**Examples:**
- `name_mismatch: false` - XMP says "John", cluster says "John" ✓
- `name_mismatch: true` - XMP says "John", cluster says "Mike" ⚠️
- `name_mismatch: null` - Only one source available
