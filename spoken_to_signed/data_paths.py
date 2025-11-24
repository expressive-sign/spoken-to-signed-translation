import os
from pathlib import Path
from typing import Optional

DATA_ROOT = Path(os.getenv("SPOKEN_TO_SIGNED_DATA_ROOT", "/media/volume/data/lexicon"))

LANGUAGE_ALIASES = {
    "ase": "asl",
    "sgg": "dgs",
    "gsg": "dgs",
    "asl": "asl",
    "dgs": "dgs",
}

SIGN_LANGUAGE_FOLDERS = {
    "asl": "ASL",
    "dgs": "DGS",
}

POSE_TYPE_DIRS = {
    "dwpose": "poses_dwpose",
    "mediapipe": "poses_mediapipe",
    "legacy": "poses",
}

DEFAULT_POSE_TYPE = "dwpose"
LEXICON_FILENAMES = ("index.csv",)


def canonical_code(code: str) -> str:
    base = (code or "").strip().lower()
    return LANGUAGE_ALIASES.get(base, base)


def dataset_folder(code: str) -> Optional[Path]:
    folder = SIGN_LANGUAGE_FOLDERS.get(canonical_code(code))
    if folder:
        return DATA_ROOT / folder
    return None


def lexicon_file(code: str) -> Optional[Path]:
    folder = dataset_folder(code)
    if not folder:
        return None
    for name in LEXICON_FILENAMES:
        candidate = folder / name
        if candidate.exists():
            return candidate
    return None


def pose_directory(code: str, pose_type: str) -> Optional[Path]:
    folder = dataset_folder(code)
    if not folder:
        return None
    dirname = POSE_TYPE_DIRS.get(pose_type, pose_type)
    candidate = folder / dirname
    if candidate.exists():
        return candidate
    return None


def pose_path(code: str, pose_type: str, video_id: str) -> Optional[Path]:
    pose_dir = pose_directory(code, pose_type)
    if not pose_dir:
        return None
    return pose_dir / f"{video_id}.pose"


def face_directory(code: str) -> Optional[Path]:
    folder = dataset_folder(code)
    if not folder:
        return None
    candidate = folder / "faces"
    if candidate.exists():
        return candidate
    return None
