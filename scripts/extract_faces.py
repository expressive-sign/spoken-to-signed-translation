#!/usr/bin/env python3
"""
Extract aligned face crops from videos and save them as MP4 files plus metadata.

Each output video keeps the same temporal stride as the pose extraction so that
face clips can later be stitched in the same order as pose segments.
"""

from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
from google.protobuf import message_factory as _message_factory
from google.protobuf import symbol_database as _symbol_database

_symbol_db = _symbol_database.Default()
if not hasattr(_symbol_db, "GetPrototype"):
    def _get_prototype(self, descriptor):
        message_class = _message_factory.GetMessageClass(descriptor)
        self.RegisterMessage(message_class)
        return message_class
    _symbol_db.__class__.GetPrototype = _get_prototype


def _expand_bbox(bbox: Tuple[float, float, float, float], margin: float,
                 width: int, height: int) -> Tuple[int, int, int, int]:
    x_min, y_min, x_max, y_max = bbox
    w = x_max - x_min
    h = y_max - y_min
    pad = margin * max(w, h)
    x_min = max(0, int(x_min - pad))
    y_min = max(0, int(y_min - pad))
    x_max = min(width, int(x_max + pad))
    y_max = min(height, int(y_max + pad))
    return x_min, y_min, x_max, y_max


def _bbox_from_landmarks(landmarks, width: int, height: int) -> Optional[Tuple[float, float, float, float]]:
    xs = [lm.x * width for lm in landmarks]
    ys = [lm.y * height for lm in landmarks]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    if x_max <= x_min or y_max <= y_min:
        return None
    return x_min, y_min, x_max, y_max


def _crop_frame(frame: np.ndarray,
                bbox: Optional[Tuple[float, float, float, float]],
                last_bbox: Optional[Tuple[float, float, float, float]],
                margin: float,
                size: int) -> Tuple[np.ndarray, Optional[Tuple[float, float, float, float]]]:
    height, width = frame.shape[:2]
    target_bbox = bbox or last_bbox
    if target_bbox is None:
        return np.zeros((size, size, 3), dtype=np.uint8), last_bbox

    x_min, y_min, x_max, y_max = _expand_bbox(target_bbox, margin, width, height)
    if x_max - x_min <= 0 or y_max - y_min <= 0:
        return np.zeros((size, size, 3), dtype=np.uint8), last_bbox

    crop = frame[y_min:y_max, x_min:x_max]
    if crop.size == 0:
        return np.zeros((size, size, 3), dtype=np.uint8), last_bbox
    resized = cv2.resize(crop, (size, size), interpolation=cv2.INTER_CUBIC)
    return resized, target_bbox


def _create_writer(path: Path, fps: float, size: int) -> Tuple[cv2.VideoWriter, str]:
    """
    Try to create an MP4 writer. Many OpenCV builds shipping with CUDA don't
    include H.264 encoders, so try mp4v first to avoid noisy FFmpeg errors.
    """
    for codec in ("mp4v", "avc1", "H264"):
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(str(path), fourcc, fps, (size, size))
        if writer.isOpened():
            return writer, codec
    raise RuntimeError("Unable to create MP4 writer with codecs mp4v/avc1/H264")


def extract_face_sequence(video_path: Path,
                          output_video: Path,
                          metadata_path: Path,
                          size: int = 256,
                          margin: float = 0.2,
                          save_every: int = 1,
                          min_confidence: float = 0.5) -> Dict:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    target_fps = source_fps / max(save_every, 1)
    output_video.parent.mkdir(parents=True, exist_ok=True)
    writer, codec_used = _create_writer(output_video, target_fps, size)

    frame_idx = 0
    saved_frames = 0
    last_bbox = None

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=min_confidence,
        min_tracking_confidence=min_confidence,
    )

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % save_every != 0:
                frame_idx += 1
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)
            bbox = None
            if results.multi_face_landmarks:
                bbox = _bbox_from_landmarks(results.multi_face_landmarks[0].landmark,
                                            frame.shape[1], frame.shape[0])

            crop, last_bbox = _crop_frame(frame, bbox, last_bbox, margin, size)
            writer.write(crop)
            saved_frames += 1
            frame_idx += 1
    finally:
        face_mesh.close()
        cap.release()
        writer.release()

    metadata = {
        "source_video": str(video_path),
        "frame_count": saved_frames,
        "fps": target_fps,
        "frame_time_ms": 1000.0 / target_fps if target_fps else 0.0,
        "source_fps": source_fps,
        "size": size,
        "margin": margin,
        "save_every": save_every,
        "codec": codec_used,
    }

    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract aligned face crops from videos.")
    parser.add_argument("--input", required=True, help="Path to a video file or a directory of videos.")
    parser.add_argument("--output", required=True,
                        help="Output MP4 path (for a single input) or directory for batch processing.")
    parser.add_argument("--size", type=int, default=256, help="Output face crop size (pixels).")
    parser.add_argument("--margin", type=float, default=0.2,
                        help="Extra margin (as ratio of face box) to include around the face.")
    parser.add_argument("--save-every", type=int, default=1,
                        help="Keep every n-th frame to match pose extraction stride.")
    parser.add_argument("--min-confidence", type=float, default=0.5,
                        help="Minimum detection confidence for MediaPipe Face Mesh.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite outputs if they already exist.")
    parser.add_argument("--jobs", type=int, default=1,
                        help="Number of videos to process in parallel (default: 1).")
    return parser.parse_args()


def iter_videos(path: Path):
    if path.is_file():
        yield path
        return
    for entry in sorted(path.iterdir()):
        if entry.is_file() and entry.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv"}:
            yield entry


def _worker_task(task, size, margin, save_every, min_confidence):
    video, out_file, meta_file = task
    metadata = extract_face_sequence(
        video,
        out_file,
        meta_file,
        size=size,
        margin=margin,
        save_every=save_every,
        min_confidence=min_confidence,
    )
    print(f"✅ {video.name}: saved {metadata['frame_count']} frames to {out_file}")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    if input_path.is_dir() and output_path.suffix:
        raise ValueError("When processing a directory, --output must be a directory path.")

    videos = list(iter_videos(input_path))
    if not videos:
        raise ValueError(f"No video files found under {input_path}")

    tasks = []
    for video in videos:
        if input_path.is_dir():
            output_path.mkdir(parents=True, exist_ok=True)
            out_file = output_path / f"{video.stem}.mp4"
        else:
            out_file = output_path

        meta_file = out_file.with_suffix(".json")
        if out_file.exists() and meta_file.exists() and not args.overwrite:
            print(f"⚠️ Face clip already exists for {video.name}, skipping.")
            continue

        tasks.append((video, out_file, meta_file))

    if not tasks:
        print("⚠️ No videos to process.")
        return

    worker = partial(
        _worker_task,
        size=args.size,
        margin=args.margin,
        save_every=args.save_every,
        min_confidence=args.min_confidence,
    )

    if args.jobs > 1 and len(tasks) > 1:
        from multiprocessing import Pool
        with Pool(processes=args.jobs) as pool:
            pool.map(worker, tasks)
    else:
        for task in tasks:
            worker(task)


if __name__ == "__main__":
    main()
