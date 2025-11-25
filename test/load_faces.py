import json
import os
import sqlite3
import sys

conn = sqlite3.connect('faces.db')
cursor = conn.cursor()

cursor.execute('''CREATE TABLE IF NOT EXISTS faces (
    image_id TEXT,
    image_path TEXT,
    person_name TEXT,
    gender TEXT,
    age INTEGER,
    confidence REAL,
    bbox_x1 REAL,
    bbox_y1 REAL,
    bbox_x2 REAL,
    bbox_y2 REAL,
    name_mismatch TEXT,
    left_eye_x REAL,
    left_eye_y REAL,
    right_eye_x REAL,
    right_eye_y REAL,
    nose_x REAL,
    nose_y REAL,
    left_mouth_x REAL,
    left_mouth_y REAL,
    right_mouth_x REAL,
    right_mouth_y REAL,
    pitch REAL,
    yaw REAL,
    roll REAL,
    cluster_id TEXT,
    cluster_name TEXT,
    cluster_confidence REAL,
    cluster_consensus_count INTEGER,
    cluster_reference_image_ids TEXT,
    cluster_is_new_cluster BOOLEAN,
    input_face_match_matched BOOLEAN,
    input_face_match_name TEXT,
    input_face_match_confidence REAL,
    input_face_match_strategy TEXT,
    input_face_match_input_bbox TEXT,
    cluster_centroid_x REAL,
    cluster_centroid_y REAL,
    input_face_match_centroid_x REAL,
    input_face_match_centroid_y REAL,
    create_time TEXT
)''')

cursor.execute('''CREATE TABLE IF NOT EXISTS unmatched_input_faces (
    image_id TEXT,
    image_path TEXT,
    name TEXT,
    x REAL,
    y REAL,
    w REAL,
    h REAL,
    centroid_x REAL,
    centroid_y REAL,
    create_time TEXT
)''')


for filename in sys.argv[1:]:
    # print(f"Processing {filename}...")
    file_mtime = os.path.getmtime(filename)
    with open(filename, 'r') as f:
        data = json.load(f)
    
    for face in data['faces']:
        lm = face['landmarks']
        pose = face['pose']
        cluster = face['cluster']
        match = face['input_face_match']
        ref_ids = cluster.get('reference_image_ids')
        
        input_bbox = match.get('input_bbox')
        cluster_centroid = cluster.get('centroid')
        match_centroid = match.get('centroid')
        cursor.execute('''INSERT INTO faces VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime(?, 'unixepoch'))''', (
            data['image_id'], data['image_path'], face.get('person_name'),
            face.get('gender'), face.get('age'), face.get('confidence'),
            face['bbox'][0], face['bbox'][1], face['bbox'][2], face['bbox'][3],
            face.get('name_mismatch'),
            lm['left_eye'][0], lm['left_eye'][1], lm['right_eye'][0], lm['right_eye'][1],
            lm['nose'][0], lm['nose'][1], lm['left_mouth'][0], lm['left_mouth'][1],
            lm['right_mouth'][0], lm['right_mouth'][1],
            pose['pitch'], pose['yaw'], pose['roll'],
            cluster['cluster_id'], cluster.get('name'), cluster.get('confidence'),
            cluster.get('consensus_count'), str(ref_ids) if ref_ids else None, cluster['is_new_cluster'],
            match['matched'], match.get('name'), match['confidence'],
            match.get('match_strategy'), str(input_bbox) if input_bbox else None,
            cluster_centroid[0] if cluster_centroid else None, cluster_centroid[1] if cluster_centroid else None,
            match_centroid[0] if match_centroid else None, match_centroid[1] if match_centroid else None,
            file_mtime
        ))

    
    for unmatched_face in data.get('unmatched_input_faces', []):
        centroid = unmatched_face.get('centroid')
        cursor.execute('''INSERT INTO unmatched_input_faces VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, datetime(?, 'unixepoch'))''', (
            data['image_id'], data['image_path'], 
            unmatched_face['name'], unmatched_face['x'], unmatched_face['y'], 
            unmatched_face['w'], unmatched_face['h'],
            centroid[0] if centroid else None, centroid[1] if centroid else None,
            file_mtime
        ))


conn.commit()
conn.close()
print("Data loaded successfully into faces.db")
