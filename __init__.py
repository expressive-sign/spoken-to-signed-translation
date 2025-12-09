import csv
import importlib
import json
import os
import sys
from functools import lru_cache
from typing import Dict, List, Sequence, Tuple
from uuid import uuid4

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
if PACKAGE_DIR not in sys.path:
    sys.path.insert(0, PACKAGE_DIR)

import numpy as np
import torch
from pose_format import Pose
from pose_format.pose_visualizer import PoseVisualizer

from .spoken_to_signed.gloss_to_pose import (
    CSVPoseLookup,
    concatenate_poses,
    gloss_to_pose as convert_gloss_to_pose,
)
from .spoken_to_signed.gloss_to_pose.lookup.fingerspelling_lookup import (
    FingerspellingPoseLookup,
)
from .spoken_to_signed.text_to_gloss.types import Gloss
from .spoken_to_signed.face_video import FaceVideoAssembler
from .spoken_to_signed.data_paths import (
    DEFAULT_POSE_TYPE,
    POSE_TYPE_DIRS,
    canonical_code,
    lexicon_file as dataset_lexicon_file,
    pose_path as dataset_pose_path,
    face_directory as dataset_face_directory,
)


AVAILABLE_GLOSSERS = ("simple", "spacylemma", "rules", "nmt")
NODE_CATEGORY = "Spoken→Signed"
ASSETS_DIR = os.path.join(PACKAGE_DIR, "assets")
DEFAULT_LEXICON = os.path.join(ASSETS_DIR, "dummy_lexicon")


def _expand_path(path: str) -> str:
    return os.path.abspath(os.path.expanduser(path))


def _parse_color(value: str) -> Tuple[int, int, int]:
    text = value.strip()
    if text.startswith("#"):
        hex_part = text[1:]
        if len(hex_part) not in (3, 6):
            raise ValueError(f"Color '{value}' must be #RGB or #RRGGBB.")
        if len(hex_part) == 3:
            hex_part = "".join(ch * 2 for ch in hex_part)
        r = int(hex_part[0:2], 16)
        g = int(hex_part[2:4], 16)
        b = int(hex_part[4:6], 16)
        return r, g, b

    parts = [p for p in text.replace(",", " ").split() if p]
    if len(parts) != 3:
        raise ValueError(
            f"Color '{value}' must be a hex string or comma-separated RGB values."
        )
    r, g, b = (int(float(p)) for p in parts)
    return tuple(max(0, min(255, c)) for c in (r, g, b))


def _load_glosser(glosser_name: str):
    if glosser_name not in AVAILABLE_GLOSSERS:
        raise ValueError(
            f"Unsupported glosser '{glosser_name}'. "
            f"Available: {', '.join(AVAILABLE_GLOSSERS)}"
        )
    return importlib.import_module(f"spoken_to_signed.text_to_gloss.{glosser_name}")


def _text_to_gloss_sentences(
    text: str,
    spoken_language: str,
    glosser_name: str,
    glosser_kwargs: Dict,
) -> List[Gloss]:
    module = _load_glosser(glosser_name)
    kwargs = dict(glosser_kwargs or {})

    try:
        sentences = module.text_to_gloss(
            text=text,
            language=spoken_language,
            **{k: v for k, v in kwargs.items() if v is not None},
        )
    except TypeError:
        # Some glossers (e.g. simple/spacylemma) do not accept signed_language.
        filtered_kwargs = {
            k: v for k, v in kwargs.items() if k not in ("signed_language",)
        }
        sentences = module.text_to_gloss(text=text, language=spoken_language, **filtered_kwargs)

    if not sentences:
        raise ValueError("Glosser returned no sentences.")

    # Normalize to List[Gloss]
    if sentences and isinstance(sentences[0], tuple):
        sentences = [sentences]  # type: ignore[assignment]

    if not isinstance(sentences, Sequence):
        raise ValueError("Unexpected glosser output. Expected sequence of gloss sentences.")

    return sentences  # type: ignore[return-value]


def _resolve_lexicon_directory(user_path: str, signed_language: str) -> str:
    if user_path and user_path.strip():
        lexicon_path = _expand_path(user_path.strip())
        if os.path.isfile(lexicon_path) or os.path.isdir(lexicon_path):
            return lexicon_path
        raise FileNotFoundError(
            f"Lexicon path '{lexicon_path}' does not exist."
        )

    dataset_csv = dataset_lexicon_file(signed_language)
    if dataset_csv:
        return str(dataset_csv)

    auto_path = _guess_lexicon_dir(signed_language)
    if auto_path is not None:
        return auto_path

    if os.path.isdir(DEFAULT_LEXICON):
        return DEFAULT_LEXICON

    raise FileNotFoundError(
        "Could not determine a lexicon directory automatically. "
        "Please provide a valid path in the node settings."
    )


@lru_cache(maxsize=None)
def _built_in_lexicons() -> List[str]:
    if not os.path.isdir(ASSETS_DIR):
        return []
    lexicons = []
    for entry in os.scandir(ASSETS_DIR):
        if entry.is_dir():
            index_path = os.path.join(entry.path, "index.csv")
            if os.path.isfile(index_path):
                lexicons.append(entry.path)
    return lexicons


@lru_cache(maxsize=None)
def _lexicon_languages(lexicon_path: str) -> List[str]:
    index_path = os.path.join(lexicon_path, "index.csv")
    languages = set()
    if os.path.isfile(index_path):
        try:
            with open(index_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    signed = row.get("signed_language")
                    if signed:
                        languages.add(signed.strip().lower())
                    if len(languages) > 32:
                        break
        except Exception:
            pass
    return sorted(languages)


def _guess_lexicon_dir(signed_language: str) -> str | None:
    code = (signed_language or "").strip().lower()
    if not code:
        return None

    preferred_names = [
        f"{code}_lexicon",
        f"{code}-lexicon",
        code,
    ]
    for lexicon_path in _built_in_lexicons():
        base = os.path.basename(lexicon_path).lower()
        if base in preferred_names or base.replace("_lexicon", "") == code:
            return lexicon_path

    for lexicon_path in _built_in_lexicons():
        if code in _lexicon_languages(lexicon_path):
            return lexicon_path
    return None


def _pose_path_builder(default_signed_language: str, default_spoken_language: str, pose_type: str):
    default_signed_code = canonical_code(default_signed_language)

    def builder(row: Dict[str, str]) -> Dict[str, str]:
        if row.get("path"):
            return row

        video_id = (row.get("video") or row.get("video_id") or "").strip()
        if not video_id:
            raise ValueError("Lexicon row is missing both 'path' and 'video'.")

        row_signed = canonical_code(row.get("signed_language") or default_signed_code)
        pose_path = dataset_pose_path(row_signed, pose_type, video_id)
        if pose_path is None:
            raise FileNotFoundError(
                f"No pose file found for signed language '{row_signed}' and video '{video_id}'."
            )

        row["path"] = str(pose_path)
        row.setdefault("signed_language", row_signed)
        row.setdefault("spoken_language", row.get("spoken_language") or default_spoken_language)

        word_value = row.get("words") or row.get("word") or row.get("gloss") or row.get("glosses") or ""
        gloss_value = row.get("glosses") or row.get("gloss") or row.get("words") or row.get("word") or ""
        row["words"] = word_value or "unknown"
        row["glosses"] = gloss_value or row["words"]

        row.setdefault("start", row.get("start") or "0")
        row.setdefault("end", row.get("end") or "0")
        row.setdefault("priority", row.get("priority") or "0")
        return row

    return builder


def _create_pose_lookup(lexicon_source: str, spoken_language: str, signed_language: str, pose_type: str, priority_mode: str = "shortest") -> CSVPoseLookup:
    source = _resolve_lexicon_directory(lexicon_source, signed_language)
    builder = _pose_path_builder(signed_language, spoken_language, pose_type)
    fingerspelling_lookup = FingerspellingPoseLookup()
    return CSVPoseLookup(source, backup=fingerspelling_lookup, pose_path_builder=builder, priority_mode=priority_mode)


def _resolve_face_dir(user_path: str, signed_language: str) -> str | None:
    if user_path and user_path.strip():
        candidate = _expand_path(user_path.strip())
        if os.path.isdir(candidate):
            return candidate
        raise FileNotFoundError(f"Face directory '{candidate}' does not exist.")

    dataset_dir = dataset_face_directory(signed_language)
    if dataset_dir:
        return str(dataset_dir)
    return None


def _gloss_to_pose_with_segments(
    gloss: Gloss,
    pose_lookup: CSVPoseLookup,
    spoken_language: str,
    signed_language: str,
):
    segments = pose_lookup.lookup_sequence(
        gloss, spoken_language, signed_language, return_metadata=True
    )
    poses = [segment[0] for segment in segments]
    pose = poses[0] if len(poses) == 1 else concatenate_poses(poses)
    return pose, segments


def _sentences_to_pose_with_segments(
    sentences: List[Gloss],
    pose_lookup: CSVPoseLookup,
    spoken_language: str,
    signed_language: str,
):
    sentence_poses: List[Pose] = []
    face_segments = []
    for gloss in sentences:
        pose, segments = _gloss_to_pose_with_segments(gloss, pose_lookup, spoken_language, signed_language)
        sentence_poses.append(pose)
        face_segments.extend(segments)

    combined = sentence_poses[0] if len(sentence_poses) == 1 else concatenate_poses(sentence_poses, trim=False)
    return combined, face_segments


def _temp_output_dir() -> str:
    path = os.path.join(PACKAGE_DIR, "generated_outputs")
    os.makedirs(path, exist_ok=True)
    return path


def _write_face_video(face_root: str, segments) -> str:
    output_path = os.path.join(_temp_output_dir(), f"faces_{uuid4().hex}.mp4")
    assembler = FaceVideoAssembler(face_root)
    assembler.write_video(segments, output_path)
    return output_path


def _sentences_to_pose(
    sentences: List[Gloss],
    pose_lookup: CSVPoseLookup,
    spoken_language: str,
    signed_language: str,
) -> "Pose":
    pose_segments = [
        convert_gloss_to_pose(gloss, pose_lookup, spoken_language, signed_language)
        for gloss in sentences
    ]
    if len(pose_segments) == 1:
        return pose_segments[0]
    return concatenate_poses(pose_segments, trim=False)


def _render_pose_frames(
    pose,
    background_color: Tuple[int, int, int],
    thickness: int,
    max_frames: int,
):
    # Fix for ZeroDivisionError in pose_visualizer.py when component.colors is empty
    for component in pose.header.components:
        if len(component.colors) == 0:
            # Assign default colors if missing (e.g. white)
            component.colors = [[255, 255, 255]]

    draw_limit = None if max_frames <= 0 else max_frames
    visualizer = PoseVisualizer(pose, thickness=thickness or None)
    frames = list(
        visualizer.draw(
            background_color=background_color,
            max_frames=draw_limit,
            transparency=False,

        )
    )
    if not frames:
        raise ValueError("Pose rendering returned no frames.")
    frames = [frame[:, :, ::-1] for frame in frames]  # BGR -> RGB
    stacked = np.stack(frames).astype(np.float32) / 255.0
    return torch.from_numpy(stacked)


class SpokenToSignedPoseVideo:
    @classmethod
    def INPUT_TYPES(cls):
        default_text = "Kleine Kinder essen Pizza in Zürich."
        return {
            "required": {
                "text": (
                    "STRING",
                    {"default": default_text, "multiline": True},
                ),
                "glosser": (AVAILABLE_GLOSSERS, {"default": "simple"}),
                "lexicon_path": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "placeholder": "Auto-select based on signed language",
                    },
                ),
                "spoken_language": (("en", "de"), {"default": "en"}),
                "signed_language": (("ase", "gsg"), {"default": "ase"}),
                "priority_selection": (("shortest", "id_gloss_max"), {"default": "shortest"}),
            },
            "optional": {
                "pose_type": (tuple(sorted(POSE_TYPE_DIRS.keys())), {"default": DEFAULT_POSE_TYPE}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "AUDIO", "VHS_VIDEOINFO")
    RETURN_NAMES = ("pose_video", "face_video", "audio", "video_info")
    FUNCTION = "generate"
    CATEGORY = NODE_CATEGORY

    def generate(
        self,
        text: str,
        glosser: str,
        lexicon_path: str,
        spoken_language: str,
        signed_language: str,
        priority_selection: str,
        pose_type: str = DEFAULT_POSE_TYPE,
    ):
        if not text.strip():
            raise ValueError("Input text must not be empty.")

        glosser_kwargs = {}
        glosser_kwargs.setdefault("signed_language", signed_language)

        sentences = _text_to_gloss_sentences(
            text=text,
            spoken_language=spoken_language,
            glosser_name=glosser,
            glosser_kwargs=glosser_kwargs,
        )

        pose_lookup = _create_pose_lookup(lexicon_path, spoken_language, signed_language, pose_type, priority_selection)
        pose, face_segments = _sentences_to_pose_with_segments(
            sentences,
            pose_lookup=pose_lookup,
            spoken_language=spoken_language,
            signed_language=signed_language,
        )

        color = _parse_color("#000000")
        images = _render_pose_frames(
            pose,
            background_color=color,
            thickness=0,
            max_frames=0,
        )

        frame_count = images.shape[0]
        height = pose.header.dimensions.height
        width = pose.header.dimensions.width
        fps = float(getattr(pose.body, "fps", 0.0) or 0.0)
        duration = frame_count / fps if fps > 0 else 0.0
        video_info = {
            "source_fps": fps,
            "source_frame_count": frame_count,
            "source_duration": duration,
            "source_width": width,
            "source_height": height,
            "loaded_fps": fps,
            "loaded_frame_count": frame_count,
            "loaded_duration": duration,
            "loaded_width": width,
            "loaded_height": height,
        }

        audio = {
            "waveform": torch.zeros(1, 1, 1),
            "sample_rate": 16000,
        }

        face_video_path = ""
        try:
            face_root = _resolve_face_dir("", signed_language)
        except FileNotFoundError:
            face_root = None
        if face_root and face_segments:
            try:
                face_video_path = _write_face_video(face_root, face_segments)
            except Exception as exc:
                face_video_path = ""
                print(f"⚠️ Failed to assemble face video: {exc}")

        return images, face_video_path, audio, video_info


NODE_CLASS_MAPPINGS = {
    "SpokenToSignedPoseVideo": SpokenToSignedPoseVideo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SpokenToSignedPoseVideo": "Spoken→Signed Pose Video",
}
