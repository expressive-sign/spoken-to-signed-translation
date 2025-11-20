import os
import sys

# Paths
raw_videos_dir = "/media/volume/data/lexicon/ASL/raw_videos"
videos_dir = "/media/volume/data/lexicon/ASL/videos"
poses_dir = "/media/volume/data/lexicon/ASL/poses"

# Threshold in bytes (10KB)
SIZE_THRESHOLD = 10 * 1024

def main():
    if not os.path.exists(raw_videos_dir):
        print(f"Error: {raw_videos_dir} not found.")
        return

    print(f"Scanning {raw_videos_dir} for files smaller than {SIZE_THRESHOLD} bytes...")

    deleted_count = 0
    processed_count = 0

    for filename in os.listdir(raw_videos_dir):
        processed_count += 1
        file_path = os.path.join(raw_videos_dir, filename)
        
        if not os.path.isfile(file_path):
            continue

        try:
            file_size = os.path.getsize(file_path)
            
            if file_size < SIZE_THRESHOLD:
                print(f"Deleting {filename} (Size: {file_size} bytes)")
                
                # Delete from raw_videos
                os.remove(file_path)
                
                # Determine video_id (filename without extension)
                video_id = os.path.splitext(filename)[0]
                
                # Delete from videos (always .mp4)
                video_output_path = os.path.join(videos_dir, f"{video_id}.mp4")
                if os.path.exists(video_output_path):
                    os.remove(video_output_path)
                    print(f"  - Deleted corresponding processed video: {video_output_path}")
                
                # Delete from poses (always .pose)
                pose_output_path = os.path.join(poses_dir, f"{video_id}.pose")
                if os.path.exists(pose_output_path):
                    os.remove(pose_output_path)
                    print(f"  - Deleted corresponding pose file: {pose_output_path}")
                
                deleted_count += 1
                
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    print(f"Scan complete.")
    print(f"Processed {processed_count} files.")
    print(f"Deleted {deleted_count} files.")

if __name__ == "__main__":
    main()
