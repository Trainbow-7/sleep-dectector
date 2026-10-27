

"""WakeWatch Video Frame Extractor.

Extracts quality, non-duplicate frames from video files for the WakeWatch dataset.

Input layout supported:
    videos/p01/awake.mp4
    videos/p01/drowsy.mp4
    videos/p01/sleeping.mp4
    (or multiple videos: videos/p01/awake/*.mp4, videos/p01/awake_01.mp4, etc.)

Output layout:
    working_frames/p01/awake/*.jpg
    working_frames/p01/drowsy/*.jpg
    working_frames/p01/sleeping/*.jpg

Usage:
    python extract_frames.py --input videos --output working_frames
    python extract_frames.py --input videos --output working_frames --interval 0.5 --max-frames 200
"""

import argparse
import sys
from pathlib import Path

import cv2
import imagehash
import pandas as pd
from PIL import Image

CLASSES = ["awake", "drowsy", "sleeping"]
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".wmv"}


def is_dark(gray, threshold=30.0):
    return float(gray.mean()) < threshold


def is_blurry(gray, threshold=20.0):
    return float(cv2.Laplacian(gray, cv2.CV_64F).var()) < threshold


def resolve_person_and_class(file_path: Path, input_root: Path):
    """Determine person_id and class label from path."""
    rel = file_path.relative_to(input_root)
    parts = rel.parts
    if len(parts) < 2:
        return None, None

    person_id = parts[0]
    # Check if parent folder or filename indicates the class
    class_name = None
    # 1. Check folder hierarchy (e.g. p01/awake/vid.mp4)
    for part in parts[1:-1]:
        part_lower = part.lower()
        for c in CLASSES:
            if part_lower == c or part_lower.startswith(c):
                class_name = c
                break
        if class_name:
            break

    # 2. Check filename if not determined from folder (e.g. p01/awake.mp4, p01/awake_01.mp4)
    if not class_name:
        stem_lower = file_path.stem.lower()
        for c in CLASSES:
            if stem_lower == c or stem_lower.startswith(c) or f"_{c}" in stem_lower:
                class_name = c
                break

    return person_id, class_name


def extract_video_frames(video_path: Path, out_dir: Path, person_id: str, class_name: str,
                         interval: float = 0.5, max_frames: int = 200,
                         blur_thresh: float = 20.0, dark_thresh: float = 30.0,
                         dup_distance: int = 5):
    """Extract filtered frames from a single video file."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [WARN] Could not open video: {video_path}")
        return 0, 0, 0, 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0 or fps > 240:
        fps = 30.0  # reasonable fallback
    step = max(1, int(round(fps * interval)))

    out_dir.mkdir(parents=True, exist_ok=True)
    video_stem = video_path.stem

    saved_count = 0
    skipped_blur = 0
    skipped_dark = 0
    skipped_dup = 0
    recent_hashes = []
    frame_idx = 0

    while cap.isOpened() and saved_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % step == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if is_dark(gray, dark_thresh):
                skipped_dark += 1
            elif is_blurry(gray, blur_thresh):
                skipped_blur += 1
            else:
                # Check duplicate against recently saved frames
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h = imagehash.phash(Image.fromarray(rgb))
                if any(h - s <= dup_distance for s in recent_hashes):
                    skipped_dup += 1
                else:
                    recent_hashes.append(h)
                    if len(recent_hashes) > 30:
                        recent_hashes.pop(0)

                    out_name = f"{person_id}_{class_name}_{video_stem}_f{frame_idx:06d}.jpg"
                    save_path = out_dir / out_name
                    cv2.imwrite(str(save_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    saved_count += 1

        frame_idx += 1

    cap.release()
    return saved_count, skipped_dark, skipped_blur, skipped_dup


def main():
    parser = argparse.ArgumentParser(description="Extract filtered frames from WakeWatch videos")
    parser.add_argument("--input", default="videos", help="Input root directory (contains p01, p02...)")
    parser.add_argument("--output", default="working_frames", help="Output directory for extracted frames")
    parser.add_argument("--interval", type=float, default=0.5, help="Time interval between frames in seconds (default: 0.5)")
    parser.add_argument("--max-frames", type=int, default=200, help="Maximum frames per video (default: 200)")
    parser.add_argument("--blur-threshold", type=float, default=20.0, help="Blur rejection threshold (default: 20.0)")
    parser.add_argument("--min-brightness", type=float, default=30.0, help="Minimum brightness threshold (default: 30.0)")
    parser.add_argument("--dup-distance", type=int, default=5, help="Phash hamming distance for duplicates (default: 5)")
    args = parser.parse_args()

    input_root = Path(args.input)
    output_root = Path(args.output)

    if not input_root.is_dir():
        sys.exit(f"Input directory does not exist: {input_root}")

    # Discover all video files
    video_files = [f for f in input_root.rglob("*") if f.is_file() and f.suffix.lower() in VIDEO_EXTS]
    if not video_files:
        sys.exit(f"No video files found inside: {input_root}")

    print(f"Found {len(video_files)} video file(s). Extracting frames (interval={args.interval}s, max={args.max_frames})...")

    records = []
    for vid in sorted(video_files):
        person_id, class_name = resolve_person_and_class(vid, input_root)
        if not person_id or not class_name:
            print(f"  [SKIP] Could not identify person and class for: {vid.relative_to(input_root)}")
            continue

        target_dir = output_root / person_id / class_name
        saved, dark, blur, dup = extract_video_frames(
            vid, target_dir, person_id, class_name,
            interval=args.interval, max_frames=args.max_frames,
            blur_thresh=args.blur_threshold, dark_thresh=args.min_brightness,
            dup_distance=args.dup_distance
        )
        records.append({
            "person": person_id,
            "class": class_name,
            "video": vid.name,
            "saved": saved,
            "dark": dark,
            "blur": blur,
            "dup": dup
        })
        print(f"  Processed {person_id}/{class_name}/{vid.name} -> {saved} frames saved (dark: {dark}, blur: {blur}, dup: {dup})")

    if not records:
        sys.exit("No valid videos processed.")

    df = pd.DataFrame(records)
    print("\n" + "=" * 50)
    print("EXTRACTION SUMMARY")
    print("=" * 50)
    summary_table = df.pivot_table(index="person", columns="class", values="saved", aggfunc="sum", fill_value=0)
    print("\nSaved frames per person and class:")
    print(summary_table.to_string())

    total_saved = df["saved"].sum()
    print(f"\nTotal frames saved: {total_saved}")
    print(f"Output saved to: {output_root}")
    print("=" * 50)


if __name__ == "__main__":
    main()
