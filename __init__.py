import csv
import importlib
import json
import os
from functools import lru_cache
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from pose_format.pose_visualizer import PoseVisualizer

from spoken_to_signed.gloss_to_pose import (
    CSVPoseLookup,
    concatenate_poses,
    gloss_to_pose as convert_gloss_to_pose,
)
from spoken_to_signed.gloss_to_pose.lookup.fingerspelling_lookup import (
    FingerspellingPoseLookup,
)
from spoken_to_signed.text_to_gloss.types import Gloss


AVAILABLE_GLOSSERS = ("simple", "spacylemma", "rules", "nmt")
NODE_CATEGORY = "Spoken→Signed"
PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(PACKAGE_DIR, "assets")
DEFAULT_LEXICON = os.path.join(ASSETS_DIR, "dummy_lexicon")


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
        lexicon_path = os.path.expanduser(user_path.strip())
        if not os.path.isdir(lexicon_path):
            raise FileNotFoundError(
                f"Lexicon directory '{lexicon_path}' does not exist."
            )
        return lexicon_path

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


def _sentences_to_pose(
    sentences: List[Gloss],
    lexicon_dir: str,
    spoken_language: str,
    signed_language: str,
) -> "Pose":
    lexicon_path = _resolve_lexicon_directory(lexicon_dir, signed_language)
    lookup = CSVPoseLookup(
        lexicon_path,
        backup=FingerspellingPoseLookup(),
    )
    pose_segments = [
        convert_gloss_to_pose(gloss, lookup, spoken_language, signed_language)
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
                "spoken_language": ("STRING", {"default": "de"}),
                "signed_language": ("STRING", {"default": "sgg"}),
                "background_color": ("STRING", {"default": "#000000"}),
            },
            "optional": {
                "glosser_options_json": ("STRING", {"default": "{}"}),
                "max_frames": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
                "thickness": ("INT", {"default": 0, "min": 0, "max": 32, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "AUDIO", "VHS_VIDEOINFO")
    RETURN_NAMES = ("images", "frame_count", "audio", "video_info")
    FUNCTION = "generate"
    CATEGORY = NODE_CATEGORY

    def generate(
        self,
        text: str,
        glosser: str,
        lexicon_path: str,
        spoken_language: str,
        signed_language: str,
        background_color: str,
        glosser_options_json: str = "{}",
        max_frames: int = 0,
        thickness: int = 0,
    ):
        if not text.strip():
            raise ValueError("Input text must not be empty.")

        try:
            glosser_kwargs = json.loads(glosser_options_json.strip() or "{}")
            if not isinstance(glosser_kwargs, dict):
                raise ValueError
        except ValueError as exc:
            raise ValueError("glosser_options_json must be a JSON object.") from exc

        glosser_kwargs.setdefault("signed_language", signed_language)

        sentences = _text_to_gloss_sentences(
            text=text,
            spoken_language=spoken_language,
            glosser_name=glosser,
            glosser_kwargs=glosser_kwargs,
        )

        pose = _sentences_to_pose(
            sentences,
            lexicon_dir=lexicon_path,
            spoken_language=spoken_language,
            signed_language=signed_language,
        )

        color = _parse_color(background_color)
        images = _render_pose_frames(
            pose,
            background_color=color,
            thickness=thickness,
            max_frames=max_frames,
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

        return images, frame_count, audio, video_info


NODE_CLASS_MAPPINGS = {
    "SpokenToSignedPoseVideo": SpokenToSignedPoseVideo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SpokenToSignedPoseVideo": "Spoken→Signed Pose Video",
}
