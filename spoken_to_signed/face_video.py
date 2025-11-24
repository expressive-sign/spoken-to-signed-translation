import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import cv2

PoseSegment = Tuple[Any, Dict[str, int]]


class FaceVideoAssembler:
    """Stitches together pre-extracted face clips following the pose sequence order."""

    def __init__(self, face_dir: str):
        self.face_dir = Path(face_dir)
        if not self.face_dir.exists():
            raise ValueError(f"Face directory {self.face_dir} does not exist")
        self._metadata_cache: Dict[str, Dict] = {}
        self._fps: float = 0.0
        self._size: Tuple[int, int] = (0, 0)

    def _load_metadata(self, video_id: str) -> Dict:
        if video_id not in self._metadata_cache:
            meta_path = self.face_dir / f"{video_id}.json"
            if not meta_path.exists():
                raise FileNotFoundError(f"Missing face metadata for {video_id} at {meta_path}")
            with meta_path.open("r", encoding="utf-8") as f:
                self._metadata_cache[video_id] = json.load(f)
        return self._metadata_cache[video_id]

    def _prepare_writer(self, metadata: Dict, output_path: Path) -> cv2.VideoWriter:
        if self._fps == 0.0:
            self._fps = metadata["fps"]
            self._size = (metadata["size"], metadata["size"])
        elif abs(self._fps - metadata["fps"]) > 1e-3:
            raise ValueError(f"Face clip FPS mismatch: expected {self._fps}, got {metadata['fps']}")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, self._fps, self._size)
        if not writer.isOpened():
            raise IOError(f"Cannot open writer at {output_path}")
        return writer

    @staticmethod
    def _frame_range(metadata: Dict, start_ms: int, end_ms: int) -> Tuple[int, int]:
        frame_time = metadata.get("frame_time_ms")
        if not frame_time:
            frame_time = 1000.0 / metadata["fps"]
        start = max(0, int(math.floor(start_ms / frame_time)))
        if end_ms <= 0:
            end = metadata["frame_count"]
        else:
            end = int(math.ceil(end_ms / frame_time))
        end = min(end, metadata["frame_count"])
        return start, max(start, end)

    def write_video(self, segments: Iterable[PoseSegment], output_path: str):
        segments = list(segments)
        if not segments:
            raise ValueError("No segments provided for face stitching")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = None

        try:
            for _, row in segments:
                video_id = Path(row["path"]).stem
                clip_path = self.face_dir / f"{video_id}.mp4"
                if not clip_path.exists():
                    raise FileNotFoundError(f"Missing face clip {clip_path}")

                metadata = self._load_metadata(video_id)
                if writer is None:
                    writer = self._prepare_writer(metadata, output_path)

                start_frame, end_frame = self._frame_range(metadata, row["start"], row["end"])
                cap = cv2.VideoCapture(str(clip_path))
                if not cap.isOpened():
                    raise IOError(f"Cannot open face clip {clip_path}")

                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                current = start_frame
                while current < end_frame:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if frame.shape[1] != self._size[0] or frame.shape[0] != self._size[1]:
                        frame = cv2.resize(frame, self._size)
                    writer.write(frame)
                    current += 1
                cap.release()
        finally:
            if writer is not None:
                writer.release()
