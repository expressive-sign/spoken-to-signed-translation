import cv2
import mediapipe as mp
import numpy as np
import argparse
import os
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

mp_holistic = mp.solutions.holistic

def extract_pose_from_video(video_path, output_path, save_every=1):
    """
    Extracts pose, hand, and face keypoints from a video and saves them as a .pose file.

    Args:
        video_path (str): path to input video (e.g. 'videos/HELLO.mp4')
        output_path (str): path to output .pose file (e.g. 'ase/HELLO.pose')
        save_every (int): keep every nth frame (to reduce file size)
    """

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    holistic = mp_holistic.Holistic(static_image_mode=False, model_complexity=2)

    frames = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % save_every != 0:
            frame_idx += 1
            continue

        # Convert frame to RGB for MediaPipe
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image_rgb)

        # Each frame: dict of landmarks
        frame_data = {}

        if results.pose_landmarks:
            frame_data["pose"] = np.array(
                [[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark]
            ).tolist()

        if results.left_hand_landmarks:
            frame_data["left_hand"] = np.array(
                [[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]
            ).tolist()

        if results.right_hand_landmarks:
            frame_data["right_hand"] = np.array(
                [[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]
            ).tolist()

        if results.face_landmarks:
            frame_data["face"] = np.array(
                [[lm.x, lm.y, lm.z] for lm in results.face_landmarks.landmark]
            ).tolist()

        frames.append(frame_data)
        frame_idx += 1

    cap.release()
    holistic.close()

    # Save all frames to .pose (JSON)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"frames": frames}, f)

    print(f"✅ Saved pose data to {output_path} with {len(frames)} frames.")


def _process_video_task(task):
    video_path, output_path, save_every = task
    try:
        extract_pose_from_video(video_path, output_path, save_every)
        return True, video_path, None
    except IOError as err:
        return False, video_path, str(err)


def process_directory(input_dir, output_dir, save_every=1, jobs=1):
    os.makedirs(output_dir, exist_ok=True)
    tasks = []

    for file in sorted(os.listdir(input_dir)):
        if file.startswith('.'):
            continue

        if not file.lower().endswith((".mp4", ".mov", ".avi")):
            continue

        input_path = os.path.join(input_dir, file)

        if not os.path.isfile(input_path):
            continue

        output_path = os.path.join(output_dir, os.path.splitext(file)[0] + ".pose")

        if os.path.exists(output_path):
            print(f"⚠️ Pose output already exists for {file}, skipping.", file=sys.stderr)
            continue

        tasks.append((os.path.abspath(input_path), os.path.abspath(output_path), save_every))

    if not tasks:
        print("⚠️ No new videos to process.", file=sys.stderr)
        return

    if jobs > 1:
        with ThreadPoolExecutor(max_workers=jobs) as executor:
            futures = [executor.submit(_process_video_task, task) for task in tasks]
            for future in as_completed(futures):
                success, _, error_msg = future.result()
                if not success and error_msg:
                    print(f"⚠️ {error_msg}", file=sys.stderr)
    else:
        for task in tasks:
            success, _, error_msg = _process_video_task(task)
            if not success and error_msg:
                print(f"⚠️ {error_msg}", file=sys.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract MediaPipe pose data from videos.")
    parser.add_argument("--input", required=True, help="Path to input video or folder")
    parser.add_argument("--output", required=True, help="Path to output .pose file or folder")
    parser.add_argument("--save_every", type=int, default=1, help="Save every n-th frame (default=1)")
    parser.add_argument("--jobs", type=int, default=1, help="Number of parallel workers to use when processing a folder (default=1)")
    args = parser.parse_args()

    # Handle single file or directory input
    if os.path.isdir(args.input):
        process_directory(args.input, args.output, args.save_every, args.jobs)
    else:
        if os.path.exists(args.output):
            print(f"⚠️ Pose output already exists for {args.input}, skipping.", file=sys.stderr)
        else:
            extract_pose_from_video(args.input, args.output, args.save_every)
