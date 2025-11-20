import csv
import json
import os
import sys

# Paths
wlasl_json_path = "../WLASL/start_kit/WLASL_v0.3.json"
poses_dir = "/media/volume/data/lexicon/ASL/poses"
output_file = "pose_gloss_mapping.csv"

def main():
    if not os.path.exists(wlasl_json_path):
        print(f"Error: {wlasl_json_path} not found.")
        return

    if not os.path.exists(poses_dir):
        print(f"Error: {poses_dir} not found.")
        return

    print(f"Reading {wlasl_json_path}...")
    with open(wlasl_json_path, 'r') as f:
        content = json.load(f)

    rows = []
    found_count = 0
    total_count = 0

    print(f"Checking poses in {poses_dir}...")
    
    for entry in content:
        gloss = entry['gloss']
        for inst in entry['instances']:
            video_id = inst['video_id']
            total_count += 1
            
            # Check if pose file exists
            pose_filename = f"{video_id}.pose"
            if os.path.exists(os.path.join(poses_dir, pose_filename)):
                rows.append((video_id, gloss))
                found_count += 1

    # Write the CSV
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["video", "gloss"])
        writer.writerows(rows)

    print(f"Processed {total_count} entries from JSON.")
    print(f"Found {found_count} matching pose files.")
    print(f"Saved mapping to {output_file}")

if __name__ == "__main__":
    main()
