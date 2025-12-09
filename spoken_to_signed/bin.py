import argparse
import importlib
import os
import tempfile
from itertools import chain
from typing import Dict, List

from pose_format import Pose

from spoken_to_signed.gloss_to_pose import gloss_to_pose, CSVPoseLookup, concatenate_poses
from spoken_to_signed.gloss_to_pose.lookup.fingerspelling_lookup import FingerspellingPoseLookup
from spoken_to_signed.face_video import FaceVideoAssembler
from spoken_to_signed.text_to_gloss.types import Gloss
from spoken_to_signed.data_paths import (
    DEFAULT_POSE_TYPE,
    POSE_TYPE_DIRS,
    canonical_code,
    lexicon_file as dataset_lexicon_file,
    pose_path as dataset_pose_path,
    face_directory as dataset_face_directory,
)


def _text_to_gloss(text: str, language: str, glosser: str, **kwargs) -> List[Gloss]:
    module = importlib.import_module(f"spoken_to_signed.text_to_gloss.{glosser}")
    return module.text_to_gloss(text=text, language=language, **kwargs)


def _expand_path(path: str) -> str:
    return os.path.abspath(os.path.expanduser(path))


def _resolve_lexicon_source(user_path: str, signed_language: str) -> str:
    if user_path and user_path.strip():
        candidate = _expand_path(user_path.strip())
        if os.path.isdir(candidate) or os.path.isfile(candidate):
            return candidate
        raise FileNotFoundError(f"Lexicon path '{candidate}' does not exist.")

    dataset_csv = dataset_lexicon_file(signed_language)
    if dataset_csv:
        return str(dataset_csv)

    return _resolve_lexicon_directory("", signed_language)


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


def _create_pose_lookup(lexicon_source: str, spoken_language: str, signed_language: str, pose_type: str) -> CSVPoseLookup:
    source = _resolve_lexicon_source(lexicon_source, signed_language)
    builder = _pose_path_builder(signed_language, spoken_language, pose_type)
    fingerspelling_lookup = FingerspellingPoseLookup()
    return CSVPoseLookup(source, backup=fingerspelling_lookup, pose_path_builder=builder)


def _resolve_face_directory(user_path: str, signed_language: str) -> str:
    if user_path and user_path.strip():
        candidate = _expand_path(user_path.strip())
        if not os.path.isdir(candidate):
            raise FileNotFoundError(f"Face directory '{candidate}' does not exist.")
        return candidate

    dataset_dir = dataset_face_directory(signed_language)
    if dataset_dir:
        return str(dataset_dir)

    raise FileNotFoundError(
        "Face directory could not be resolved automatically. "
        "Please provide --face-dir explicitly."
    )


def _gloss_to_pose(sentences: List[Gloss], pose_lookup: CSVPoseLookup, spoken_language: str, signed_language: str) -> Pose:
    poses = [gloss_to_pose(gloss, pose_lookup, spoken_language, signed_language) for gloss in sentences]
    if len(poses) == 1:
        return poses[0]
    return concatenate_poses(poses, trim=False)


def _gloss_to_pose_with_segments(gloss: Gloss,
                                 pose_lookup: CSVPoseLookup,
                                 spoken_language: str,
                                 signed_language: str):
    segments = pose_lookup.lookup_sequence(
        gloss, spoken_language, signed_language, return_metadata=True)
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


def _get_models_dir():
    home_dir = os.path.expanduser("~")
    sign_dir = os.path.join(home_dir, ".sign")
    os.makedirs(sign_dir, exist_ok=True)
    models_dir = os.path.join(sign_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    return models_dir


def _pose_to_video(pose: Pose, video_path: str):
    models_dir = _get_models_dir()
    pix2pix_path = os.path.join(models_dir, "pix2pix.h5")
    if not os.path.exists(pix2pix_path):
        print("Downloading pix2pix model")
        import urllib.request
        urllib.request.urlretrieve(
            "https://firebasestorage.googleapis.com/v0/b/sign-mt-assets/o/models%2Fgenerator%2Fmodel.h5?alt=media",
            pix2pix_path)

    import subprocess

    try:
        subprocess.run(["command", "-v", "pose_to_video"], shell=True, check=True)
    except subprocess.CalledProcessError:
        raise RuntimeError(
            "The command 'pose_to_video' does not exist. Please install the `transcription` package using "
            "`pip install git+https://github.com/sign-language-processing/transcription`")

    pose_path = tempfile.mktemp(suffix=".pose")
    with open(pose_path, "wb") as f:
        pose.write(f)

    args = ["pose_to_video", "--type=pix_to_pix",
            "--model", pix2pix_path,
            "--pose", pose_path,
            "--video", video_path,
            "--upscale"]
    print(" ".join(args))
    subprocess.run(args, shell=True, check=True)


def _text_input_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--glosser", choices=['simple', 'spacylemma', 'rules', 'nmt'], required=True)

    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--lexicon", type=str, default="")
    pre_args, _ = pre_parser.parse_known_args()

    if pre_args.lexicon:
        lookup = CSVPoseLookup(pre_args.lexicon)
        spoken_languages = sorted(list(lookup.words_index.keys()))
        signed_languages = sorted(set(chain.from_iterable(lookup.words_index[lang].keys() for lang in spoken_languages)))
    else:
        spoken_languages = ['en', 'de']
        signed_languages = ['ase', 'gsg']

    parser.add_argument("--spoken-language", choices=spoken_languages, required=True)
    parser.add_argument("--signed-language", choices=signed_languages, required=True)
    parser.add_argument("--lexicon", type=str, default="", help="Optional path to a lexicon directory or CSV file.")
    parser.add_argument(
        "--pose-type",
        choices=sorted(POSE_TYPE_DIRS.keys()),
        default=DEFAULT_POSE_TYPE,
        help="Choose which pose dataset variant to use.",
    )


def text_to_gloss():
    args_parser = argparse.ArgumentParser()
    _text_input_arguments(args_parser)
    args = args_parser.parse_args()

    print("Text to gloss")
    print("Input text:", args.text)
    sentences = _text_to_gloss(args.text, args.spoken_language, args.glosser)
    print("Output gloss:", sentences)


def pose_to_video():
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--pose", type=str, required=True)
    args_parser.add_argument("--video", type=str, required=True)
    args = args_parser.parse_args()

    with open(args.pose, "rb") as f:
        pose = Pose.read(f.read())

    _pose_to_video(pose, args.video)

    print("Pose to video")
    print("Input pose:", args.pose)
    print("Output video:", args.video)


def text_to_gloss_to_pose():
    args_parser = argparse.ArgumentParser()
    _text_input_arguments(args_parser)
    args_parser.add_argument("--pose", type=str, required=True)
    args = args_parser.parse_args()

    sentences = _text_to_gloss(args.text, args.spoken_language, args.glosser)
    pose_lookup = _create_pose_lookup(args.lexicon, args.spoken_language, args.signed_language, args.pose_type)
    pose = _gloss_to_pose(sentences, pose_lookup, args.spoken_language, args.signed_language)

    with open(args.pose, "wb") as f:
        pose.write(f)

    print("Text to gloss to pose")
    print("Input text:", args.text)
    print("Output pose:", args.pose)


def text_to_gloss_to_pose_to_video():
    args_parser = argparse.ArgumentParser()
    _text_input_arguments(args_parser)
    args_parser.add_argument("--video", type=str, required=True)
    args = args_parser.parse_args()

    sentences = _text_to_gloss(args.text, args.spoken_language, args.glosser, signed_language=args.signed_language)
    pose_lookup = _create_pose_lookup(args.lexicon, args.spoken_language, args.signed_language, args.pose_type)
    pose = _gloss_to_pose(sentences, pose_lookup, args.spoken_language, args.signed_language)
    _pose_to_video(pose, args.video)

    print("Text to gloss to pose to video")
    print("Input text:", args.text)
    print("Output video:", args.video)


def text_to_gloss_to_pose_and_faces():
    args_parser = argparse.ArgumentParser()
    _text_input_arguments(args_parser)
    args_parser.add_argument("--pose", type=str, required=True)
    args_parser.add_argument("--face-dir", type=str, default="",
                             help="Directory containing per-video face crops (optional; auto-detected if empty).")
    args_parser.add_argument("--face-video", type=str, required=True,
                             help="Output path for the stitched face video.")
    args = args_parser.parse_args()

    sentences = _text_to_gloss(args.text, args.spoken_language, args.glosser,
                               signed_language=args.signed_language)

    pose_lookup = _create_pose_lookup(args.lexicon, args.spoken_language, args.signed_language, args.pose_type)
    pose, face_segments = _sentences_to_pose_with_segments(
        sentences,
        pose_lookup,
        args.spoken_language,
        args.signed_language,
    )

    with open(args.pose, "wb") as f:
        pose.write(f)

    face_dir = _resolve_face_directory(args.face_dir, args.signed_language)
    assembler = FaceVideoAssembler(face_dir)
    assembler.write_video(face_segments, args.face_video)

    print("Text to gloss to pose and face video")
    print("Input text:", args.text)
    print("Pose output:", args.pose)
    print("Face video output:", args.face_video)


if __name__ == "__main__":
    text_to_gloss_to_pose()
