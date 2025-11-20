import argparse
import csv
from pathlib import Path

from pose_format import Pose


def pose_length(pose_path: Path) -> int:
    """Return number of frames in a pose-format file."""
    with open(pose_path, "rb") as f:
        pose = Pose.read(f.read())
    return len(pose.body.data)


def build_index(mapping_path: Path, pose_dir: Path, output_path: Path,
                spoken_language: str, signed_language: str):
    rows_by_gloss = {}
    total = 0
    missing = 0

    with mapping_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            pose_path = pose_dir / f"{row['video']}.pose"
            if not pose_path.exists():
                missing += 1
                continue

            length = pose_length(pose_path)
            rows_by_gloss.setdefault(row["gloss"], []).append((length, pose_path))

    # Sort poses for each gloss by length (shortest first) and assign priority accordingly.
    output_rows = []
    for gloss, items in rows_by_gloss.items():
        items.sort(key=lambda x: x[0])
        for priority, (_, pose_path) in enumerate(items):
            output_rows.append({
                "path": str(pose_path),
                "spoken_language": spoken_language,
                "signed_language": signed_language,
                "start": 0,
                "end": 0,
                "words": gloss,
                "glosses": gloss,
                "priority": priority,
            })

    fields = ["path", "spoken_language", "signed_language", "start", "end", "words", "glosses", "priority"]
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"Wrote {len(output_rows)} rows to {output_path}")
    print(f"Missing {missing} pose files out of {total} mappings")


def main():
    parser = argparse.ArgumentParser(description="Rebuild ASL lexicon index with shortest poses prioritized.")
    parser.add_argument("--mapping", type=Path, default=Path("scripts/video_gloss_mapping.csv"),
                        help="CSV mapping of video id to gloss")
    parser.add_argument("--pose-dir", type=Path, default=Path("/media/volume/data/lexicon/ASL/poses"),
                        help="Directory containing pose-format files")
    parser.add_argument("--output", type=Path, default=Path("assets/asl_lexicon/index.csv"),
                        help="Output index.csv path")
    parser.add_argument("--spoken-language", default="en")
    parser.add_argument("--signed-language", default="ase")
    args = parser.parse_args()

    build_index(args.mapping, args.pose_dir, args.output, args.spoken_language, args.signed_language)


if __name__ == "__main__":
    main()
