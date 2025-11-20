import csv
import json
import os
import sys

# Paths
wlasl_json_path = "../WLASL/start_kit/WLASL_v0.3.json"
video_dir = "/media/volume/data/lexicon/ASL/videos"
output_file = "video_gloss_mapping.csv"

def main():
    if not os.path.exists(wlasl_json_path):
        print(f"Error: {wlasl_json_path} not found.")
        return

    if not os.path.exists(video_dir):
        print(f"Error: {video_dir} not found.")
        return

    print(f"Reading {wlasl_json_path}...")
    with open(wlasl_json_path, 'r') as f:
        content = json.load(f)

    rows = []
    found_count = 0
    total_count = 0

    print(f"Checking videos in {video_dir}...")
    
    for entry in content:
        gloss = entry['gloss']
        for inst in entry['instances']:
            video_id = inst['video_id']
            total_count += 1
            
            # Check if video file exists (processed videos are always .mp4)
            found = False
            for ext in ['.mp4']:
                if os.path.exists(os.path.join(video_dir, f"{video_id}{ext}")):
                    found = True
                    break
            
            if found:
                rows.append((video_id, gloss))
                found_count += 1

    # Write the CSV
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["video", "gloss"])
        writer.writerows(rows)

    print(f"Processed {total_count} videos.")
    print(f"Found {found_count} videos.")
    print(f"Saved mapping to {output_file}")

if __name__ == "__main__":
    main()
