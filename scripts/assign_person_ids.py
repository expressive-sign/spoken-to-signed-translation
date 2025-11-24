import face_recognition
import cv2
import pandas as pd
import os
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

VIDEO_DIR = "/media/volume/data/lexicon/ASL/videos"
INDEX_PATH = "assets/asl_lexicon/index.csv"
OUTPUT_PATH = "assets/asl_lexicon/index.csv"

def process_video(video_id):
    video_filename = f"{video_id}.mp4"
    video_path = os.path.join(VIDEO_DIR, video_filename)
    
    if not os.path.exists(video_path):
        return video_id, None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return video_id, None
    
    # Try to grab a frame from the middle
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count // 2)
    
    ret, frame = cap.read()
    if not ret:
        # Try beginning if middle fails
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()
    
    cap.release()
    
    if not ret:
        return video_id, None

    # Convert BGR to RGB (face_recognition uses RGB)
    rgb_frame = frame[:, :, ::-1]
    rgb_frame = np.ascontiguousarray(rgb_frame)
    
    # Detect faces
    face_locations = face_recognition.face_locations(rgb_frame)
    if not face_locations:
        return video_id, None
        
    # Get encodings (assume the first face is the signer)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    
    if face_encodings:
        return video_id, face_encodings[0]
    
    return video_id, None

def main():
    print(f"Reading {INDEX_PATH}...")
    df = pd.read_csv(INDEX_PATH, dtype={'video': str})
    unique_videos = df['video'].unique()
    
    print(f"Processing {len(unique_videos)} unique videos with {multiprocessing.cpu_count()} workers...")
    
    video_encodings = {}
    
    # Phase 1: Extract encodings in parallel
    with ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(process_video, unique_videos), total=len(unique_videos)))
        
    for video_id, encoding in results:
        video_encodings[video_id] = encoding

    # Phase 2: Assign IDs sequentially
    print("Assigning IDs...")
    known_encodings = []
    known_ids = []
    video_to_person_id = {}
    
    for video_id in unique_videos:
        encoding = video_encodings.get(video_id)
        
        if encoding is None:
            video_to_person_id[video_id] = -1
            continue
            
        # Compare with known faces
        matches = face_recognition.compare_faces(known_encodings, encoding, tolerance=0.6)
        
        person_id = -1
        if True in matches:
            first_match_index = matches.index(True)
            person_id = known_ids[first_match_index]
        else:
            # New person
            new_id = len(known_ids)
            known_encodings.append(encoding)
            known_ids.append(new_id)
            person_id = new_id
            
        video_to_person_id[video_id] = person_id

    # Map back to dataframe
    print("Updating dataframe...")
    df['person_id'] = df['video'].map(video_to_person_id)
    df['person_id'] = df['person_id'].fillna(-1).astype(int)
    
    print(f"Saving to {OUTPUT_PATH}...")
    df.to_csv(OUTPUT_PATH, index=False)
    print("Done.")

if __name__ == "__main__":
    main()
