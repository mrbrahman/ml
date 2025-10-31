import os
import json
from pathlib import Path
from models import model_manager
from vector_store import vector_store
from typing import Dict, List

class FaceTrainer:
    def __init__(self, training_data_path: str = "training_data"):
        self.training_data_path = Path(training_data_path)
        self.training_data_path.mkdir(exist_ok=True)
    
    def train_from_dataset(self, dataset_path: str):
        """
        Train face recognition from a dataset with structure:
        dataset_path/
        ├── person1/
        │   ├── image1.jpg
        │   ├── image2.jpg
        └── person2/
            ├── image1.jpg
            └── image2.jpg
        """
        dataset_path = Path(dataset_path)
        if not dataset_path.exists():
            print(f"Training dataset not found: {dataset_path}")
            return False
        
        print(f"Training face recognition from: {dataset_path}")
        trained_faces = 0
        
        for person_dir in dataset_path.iterdir():
            if not person_dir.is_dir():
                continue
                
            person_name = person_dir.name
            print(f"Processing {person_name}...")
            
            # Process all images for this person
            person_embeddings = []
            for image_file in person_dir.glob("*.{jpg,jpeg,png,bmp}"):
                try:
                    faces_data = model_manager.detect_faces(str(image_file))
                    if faces_data:
                        # Use the first/largest face found
                        face_data = max(faces_data, key=lambda x: x['confidence'])
                        person_embeddings.append(face_data['embedding'])
                        print(f"  ✅ {image_file.name}")
                    else:
                        print(f"  ❌ No face found in {image_file.name}")
                except Exception as e:
                    print(f"  ❌ Error processing {image_file.name}: {e}")
            
            # Create cluster for this person if we have embeddings
            if person_embeddings:
                # Use first embedding to create cluster, then add others
                cluster_id, _ = vector_store.add_face_embedding(
                    f"training_{person_name}_0", 
                    person_embeddings[0]
                )
                
                # Add remaining embeddings to the same cluster
                for i, embedding in enumerate(person_embeddings[1:], 1):
                    vector_store.add_face_embedding(
                        f"training_{person_name}_{i}", 
                        embedding
                    )
                
                # Name the cluster
                vector_store.name_face_cluster(cluster_id, person_name)
                trained_faces += len(person_embeddings)
                print(f"  ✅ Created cluster '{cluster_id}' for {person_name} with {len(person_embeddings)} faces")
        
        print(f"Training complete! Processed {trained_faces} faces")
        return True
    
    def train_from_json(self, json_path: str):
        """
        Train from JSON file with format:
        {
          "person1": ["/path/to/image1.jpg", "/path/to/image2.jpg"],
          "person2": ["/path/to/image3.jpg"]
        }
        """
        json_path = Path(json_path)
        if not json_path.exists():
            print(f"Training JSON not found: {json_path}")
            return False
        
        with open(json_path, 'r') as f:
            training_data = json.load(f)
        
        print(f"Training face recognition from: {json_path}")
        trained_faces = 0
        
        for person_name, image_paths in training_data.items():
            print(f"Processing {person_name}...")
            
            person_embeddings = []
            for image_path in image_paths:
                if not os.path.exists(image_path):
                    print(f"  ❌ Image not found: {image_path}")
                    continue
                
                try:
                    faces_data = model_manager.detect_faces(image_path)
                    if faces_data:
                        face_data = max(faces_data, key=lambda x: x['confidence'])
                        person_embeddings.append(face_data['embedding'])
                        print(f"  ✅ {Path(image_path).name}")
                    else:
                        print(f"  ❌ No face found in {Path(image_path).name}")
                except Exception as e:
                    print(f"  ❌ Error processing {Path(image_path).name}: {e}")
            
            if person_embeddings:
                cluster_id, _ = vector_store.add_face_embedding(
                    f"training_{person_name}_0", 
                    person_embeddings[0]
                )
                
                for i, embedding in enumerate(person_embeddings[1:], 1):
                    vector_store.add_face_embedding(
                        f"training_{person_name}_{i}", 
                        embedding
                    )
                
                vector_store.name_face_cluster(cluster_id, person_name)
                trained_faces += len(person_embeddings)
                print(f"  ✅ Created cluster '{cluster_id}' for {person_name} with {len(person_embeddings)} faces")
        
        print(f"Training complete! Processed {trained_faces} faces")
        return True

# Global trainer instance
face_trainer = FaceTrainer()