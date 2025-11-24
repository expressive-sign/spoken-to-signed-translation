#!/usr/bin/env python3
"""
Extract pose keypoints from videos using the DWPose detector and store them in
the same JSON layout produced by scripts/extract_poses.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import torch
from pose_format import Pose
from pose_format.numpy import NumPyPoseBody
from pose_format.pose_header import PoseHeader, PoseHeaderComponent, PoseHeaderDimensions

# Allow importing comfyui_controlnet_aux without requiring it to be installed as a package
SCRIPT_PATH = Path(__file__).resolve()
CUSTOM_NODES_PATH = SCRIPT_PATH.parents[2]
if str(CUSTOM_NODES_PATH) not in sys.path:
    sys.path.insert(0, str(CUSTOM_NODES_PATH))

from comfyui_controlnet_aux.src.custom_controlnet_aux.dwpose import DwposeDetector  # type: ignore
from comfyui_controlnet_aux.src.custom_controlnet_aux.util import DWPOSE_MODEL_NAME  # type: ignore


def _resolve_bbox_repo(detector_name: str) -> str:
    if detector_name == "None" or detector_name == "yolox_l.onnx":
        return DWPOSE_MODEL_NAME
    if "yolox" in detector_name:
        return "hr16/yolox-onnx"
    if "yolo_nas" in detector_name:
        return "hr16/yolo-nas-fp16"
    raise ValueError(f"Unsupported bbox detector: {detector_name}")


def _resolve_pose_repo(model_name: str) -> str:
    if model_name in {"dw-ll_ucoco_384.onnx", "dw-ll_ucoco.onnx"}:
        return DWPOSE_MODEL_NAME
    if model_name.endswith(".onnx"):
        return "hr16/UnJIT-DWPose"
    if model_name.endswith(".torchscript.pt"):
        return "hr16/DWPose-TorchScript-BatchSize5"
    raise ValueError(f"Unsupported pose estimator: {model_name}")


def _convert_keypoints(
    data: Optional[List[float]],
    dims: int,
) -> List[List[float]]:
    if not data:
        return []

    converted: List[List[float]] = []
    for i in range(0, len(data), 3):
        x = float(data[i])
        y = float(data[i + 1])
        confidence = float(data[i + 2])
        if dims == 4:
            converted.append([x, y, 0.0, confidence])
        else:
            converted.append([x, y, 0.0])
    return converted


BODY_POINT_MAP = [
    ("NOSE", 0),
    ("NECK", 1),
    ("RIGHT_SHOULDER", 2),
    ("RIGHT_ELBOW", 3),
    ("RIGHT_WRIST", 4),
    ("LEFT_SHOULDER", 5),
    ("LEFT_ELBOW", 6),
    ("LEFT_WRIST", 7),
    ("MID_HIP", 8),
    ("RIGHT_HIP", 9),
    ("RIGHT_KNEE", 10),
    ("RIGHT_ANKLE", 11),
    ("LEFT_HIP", 12),
    ("LEFT_KNEE", 13),
    ("LEFT_ANKLE", 14),
    ("RIGHT_EYE", 15),
    ("LEFT_EYE", 16),
    ("RIGHT_EAR", 17),
    ("LEFT_EAR", 18),
    ("LEFT_BIG_TOE", 19),
    ("LEFT_SMALL_TOE", 20),
    ("LEFT_HEEL", 21),
    ("RIGHT_BIG_TOE", 22),
    ("RIGHT_SMALL_TOE", 23),
    ("RIGHT_HEEL", 24),
]

HAND_POINT_NAMES = [
    "WRIST",
    "THUMB_CMC",
    "THUMB_MCP",
    "THUMB_IP",
    "THUMB_TIP",
    "INDEX_FINGER_MCP",
    "INDEX_FINGER_PIP",
    "INDEX_FINGER_DIP",
    "INDEX_FINGER_TIP",
    "MIDDLE_FINGER_MCP",
    "MIDDLE_FINGER_PIP",
    "MIDDLE_FINGER_DIP",
    "MIDDLE_FINGER_TIP",
    "RING_FINGER_MCP",
    "RING_FINGER_PIP",
    "RING_FINGER_DIP",
    "RING_FINGER_TIP",
    "PINKY_MCP",
    "PINKY_PIP",
    "PINKY_DIP",
    "PINKY_TIP",
]

FACE_POINT_COUNT = 70
FACE_POINT_NAMES = [f"FACE_{i}" for i in range(FACE_POINT_COUNT)]


def _split_points(raw: Optional[List[List[float]]], values_per_point: int) -> List[List[float]]:
    if not raw:
        return []
    # raw already grouped lists of length values? ensures dims.
    return raw


def _component_arrays(
    frames: List[Dict[str, List[List[float]]]],
    key: str,
    point_map: List[Tuple[str, int]],
    values_per_point: int,
    has_confidence: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    frame_count = len(frames)
    points_count = len(point_map)
    data = np.zeros((frame_count, 1, points_count, 3), dtype=np.float32)
    confidence = np.zeros((frame_count, 1, points_count), dtype=np.float32)

    for frame_idx, frame_data in enumerate(frames):
        point_values = frame_data.get(key)
        if not point_values:
            continue
        for target_idx, (_, source_idx) in enumerate(point_map):
            if source_idx >= len(point_values):
                continue
            point = point_values[source_idx]
            x, y = point[0], point[1]
            data[frame_idx, 0, target_idx] = (x, y, 0.0)
            conf = point[3] if has_confidence and len(point) > 3 else 1.0
            confidence[frame_idx, 0, target_idx] = conf

    return data, confidence


def _generic_component_arrays(
    frames: List[Dict[str, List[List[float]]]],
    key: str,
    point_names: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    frame_count = len(frames)
    points_count = len(point_names)
    data = np.zeros((frame_count, 1, points_count, 3), dtype=np.float32)
    confidence = np.zeros((frame_count, 1, points_count), dtype=np.float32)

    for frame_idx, frame_data in enumerate(frames):
        point_values = frame_data.get(key)
        if not point_values:
            continue
        for point_idx in range(min(points_count, len(point_values))):
            point = point_values[point_idx]
            x, y = point[0], point[1]
            data[frame_idx, 0, point_idx] = (x, y, 0.0)
            confidence[frame_idx, 0, point_idx] = 1.0

    return data, confidence


def frames_to_pose(
    frames: List[Dict[str, List[List[float]]]],
    width: int,
    height: int,
    fps: float,
    include_body: bool,
    include_hands: bool,
    include_face: bool,
) -> Pose:
    components: List[PoseHeaderComponent] = []
    data_arrays: List[np.ndarray] = []
    confidence_arrays: List[np.ndarray] = []

    if include_body:
        body_data, body_conf = _component_arrays(frames, "pose", BODY_POINT_MAP, values_per_point=4, has_confidence=True)
        components.append(
            PoseHeaderComponent(
                name="POSE_LANDMARKS",
                points=[name for name, _ in BODY_POINT_MAP],
                limbs=[],
                colors=[],
                point_format="XYZC",
            )
        )
        data_arrays.append(body_data)
        confidence_arrays.append(body_conf)

    if include_hands:
        left_data, left_conf = _generic_component_arrays(frames, "left_hand", HAND_POINT_NAMES)
        right_data, right_conf = _generic_component_arrays(frames, "right_hand", HAND_POINT_NAMES)
        components.append(
            PoseHeaderComponent(
                name="LEFT_HAND_LANDMARKS",
                points=HAND_POINT_NAMES,
                limbs=[],
                colors=[],
                point_format="XYZC",
            )
        )
        components.append(
            PoseHeaderComponent(
                name="RIGHT_HAND_LANDMARKS",
                points=HAND_POINT_NAMES,
                limbs=[],
                colors=[],
                point_format="XYZC",
            )
        )
        data_arrays.extend([left_data, right_data])
        confidence_arrays.extend([left_conf, right_conf])

    if include_face:
        face_data, face_conf = _generic_component_arrays(frames, "face", FACE_POINT_NAMES)
        components.append(
            PoseHeaderComponent(
                name="FACE_LANDMARKS",
                points=FACE_POINT_NAMES,
                limbs=[],
                colors=[],
                point_format="XYZC",
            )
        )
        data_arrays.append(face_data)
        confidence_arrays.append(face_conf)

    if not components:
        raise ValueError("No components enabled for extraction.")

    total_data = np.concatenate(data_arrays, axis=2)
    total_confidence = np.concatenate(confidence_arrays, axis=2)

    pose_body = NumPyPoseBody(fps=fps, data=total_data, confidence=total_confidence)
    header = PoseHeader(
        version=0.2,
        dimensions=PoseHeaderDimensions(width=float(width), height=float(height), depth=0.0),
        components=components,
        is_bbox=False,
    )
    return Pose(header=header, body=pose_body)


class DWPoseExtractor:
    def __init__(
        self,
        detector: DwposeDetector,
        include_body: bool,
        include_hands: bool,
        include_face: bool,
    ) -> None:
        self.detector = detector
        self.include_body = include_body
        self.include_hands = include_hands
        self.include_face = include_face

    def _frame_to_pose(self, frame_bgr) -> Dict[str, List[List[float]]]:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        _, openpose_dict = self.detector(
            frame_rgb,
            include_body=self.include_body,
            include_hand=self.include_hands,
            include_face=self.include_face,
            image_and_json=True,
            output_type="np",
        )
        people = openpose_dict.get("people", [])
        width = int(openpose_dict.get("canvas_width", frame_rgb.shape[1]))
        height = int(openpose_dict.get("canvas_height", frame_rgb.shape[0]))

        if not people:
            return {}

        person = people[0]
        frame_data: Dict[str, List[List[float]]] = {}

        pose = _convert_keypoints(person.get("pose_keypoints_2d"), dims=4)
        if pose:
            frame_data["pose"] = pose

        left_hand = _convert_keypoints(
            person.get("hand_left_keypoints_2d"), dims=3
        )
        if left_hand:
            frame_data["left_hand"] = left_hand

        right_hand = _convert_keypoints(
            person.get("hand_right_keypoints_2d"), dims=3
        )
        if right_hand:
            frame_data["right_hand"] = right_hand

        face = _convert_keypoints(
            person.get("face_keypoints_2d"), dims=3
        )
        if face:
            frame_data["face"] = face

        return frame_data

    def extract_from_video(self, video_path: Path, output_path: Path, save_every: int) -> None:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

        frames: List[Dict[str, List[List[float]]]] = []
        idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % save_every == 0:
                frame_data = self._frame_to_pose(frame)
                frames.append(frame_data)
            idx += 1

        cap.release()

        pose = frames_to_pose(
            frames,
            width=width,
            height=height,
            fps=fps,
            include_body=self.include_body,
            include_hands=self.include_hands,
            include_face=self.include_face,
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wb") as f:
            pose.write(f)

        print(f"✅ Saved {len(frames)} frames to {output_path}")


def iter_video_files(input_path: Path) -> Iterable[Path]:
    if input_path.is_file():
        yield input_path
        return

    for file in sorted(input_path.iterdir()):
        if not file.is_file():
            continue
        if file.suffix.lower() not in {".mp4", ".mov", ".avi", ".mkv"}:
            continue
        yield file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract DWPose keypoints from a video or directory of videos."
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of videos to process in parallel when --input is a folder.",
    )
    parser.add_argument("--input", required=True, help="Path to a video file or folder of videos.")
    parser.add_argument(
        "--output",
        required=True,
        help="Output .pose file (for single input) or folder (for directory input).",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=1,
        help="Keep every n-th frame when writing poses (default: 1).",
    )
    parser.add_argument(
        "--bbox-detector",
        default="yolox_l.onnx",
        help="YOLO model file name for bounding box detection.",
    )
    parser.add_argument(
        "--pose-estimator",
        default="dw-ll_ucoco_384.onnx",
        help="Pose model file name for DWPose.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device to run TorchScript weights on (default: cuda if available).",
    )
    parser.add_argument(
        "--disable-hands",
        action="store_true",
        help="Skip extracting hand keypoints.",
    )
    parser.add_argument(
        "--disable-face",
        action="store_true",
        help="Skip extracting face keypoints.",
    )
    parser.add_argument(
        "--disable-body",
        action="store_true",
        help="Skip extracting body keypoints.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite pose files if they already exist when processing a directory.",
    )
    return parser.parse_args()


def _load_detector(args: argparse.Namespace) -> DWPoseExtractor:
    pose_repo = _resolve_pose_repo(args.pose_estimator)
    bbox_repo = (
        _resolve_bbox_repo(args.bbox_detector)
        if args.bbox_detector != "None"
        else DWPOSE_MODEL_NAME
    )

    detector = DwposeDetector.from_pretrained(
        pose_repo,
        bbox_repo,
        det_filename=(None if args.bbox_detector == "None" else args.bbox_detector),
        pose_filename=args.pose_estimator,
        torchscript_device=args.device,
    )
    return DWPoseExtractor(
        detector,
        include_body=not args.disable_body,
        include_hands=not args.disable_hands,
        include_face=not args.disable_face,
    )


def _process_video(video_file: Path, pose_file: Path, save_every: int, extractor: DWPoseExtractor):
    extractor.extract_from_video(video_file, pose_file, save_every)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.save_every < 1:
        raise ValueError("--save-every must be >= 1")

    if input_path.is_dir() and output_path.suffix:
        raise ValueError("When --input is a directory, --output must be a directory as well.")

    extractor = _load_detector(args)

    if input_path.is_file():
        extractor.extract_from_video(input_path, output_path, args.save_every)
        return

    output_path.mkdir(parents=True, exist_ok=True)

    videos = list(iter_video_files(input_path))
    if not videos:
        print("⚠️ No videos found to process.")
        return

    tasks: List[Tuple[Path, Path]] = []
    for video_file in videos:
        pose_file = output_path / f"{video_file.stem}.pose"
        if pose_file.exists() and not args.overwrite:
            print(f"⚠️ {pose_file} exists, skipping.")
            continue
        tasks.append((video_file, pose_file))

    if not tasks:
        print("⚠️ All pose files already exist. Use --overwrite to regenerate.")
        return

    if args.jobs > 1 and len(tasks) > 1:
        with ThreadPoolExecutor(max_workers=args.jobs) as executor:
            futures = {
                executor.submit(_process_video, video, pose, args.save_every, extractor): (video, pose)
                for video, pose in tasks
            }
            for future in as_completed(futures):
                video, pose = futures[future]
                try:
                    future.result()
                except Exception as err:  # pragma: no cover
                    print(f"❌ Failed to process {video}: {err}")
    else:
        for video, pose in tasks:
            try:
                extractor.extract_from_video(video, pose, args.save_every)
            except Exception as err:  # pragma: no cover
                print(f"❌ Failed to process {video}: {err}")


if __name__ == "__main__":
    main()
