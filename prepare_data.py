"""WakeWatch dataset preparation.

Input layout (your folders):
    <input>/p01/awake/*.jpg   <input>/p01/drowsy/*.jpg   <input>/p01/sleeping/*.jpg
    <input>/p02/...           ...                        <input>/p09/...

What it does, in order:
    1. Reads every image (unreadable files are rejected)
    2. Finds and crops the face (with margin) and resizes to 224x224
    3. Rejects blurry, too dark and overexposed crops
    4. Removes near-duplicates (same person and class only)
    5. Splits by PERSON into train / val / test (a person never crosses splits)
    6. Writes labels.csv, a summary report, sample images and clean.zip

Your original photos are never modified or deleted. Rejected images are COPIED
to <output>/_rejected/<reason>/ so you can review them and rescue any you want.

Usage:
    python prepare_data.py --input working --output data
    python prepare_data.py --input working --output data --keep-no-face
"""

import argparse
import itertools
import random
import shutil
import sys
import zipfile
from pathlib import Path

import cv2
import imagehash
import numpy as np
import pandas as pd
from PIL import Image

CLASSES = ["awake", "drowsy", "sleeping"]
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SIZE = 224


# ----------------------------------------------------------------- face detection
class FaceCropper:
    """Uses MediaPipe if it works, otherwise OpenCV's Haar cascade."""

    def __init__(self):
        self.kind = None
        try:
            import mediapipe as mp

            self._mp_det = mp.solutions.face_detection.FaceDetection(
                model_selection=1, min_detection_confidence=0.4
            )
            self.kind = "mediapipe"
            return
        except Exception:
            pass
        try:
            path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            cascade = cv2.CascadeClassifier(path)
            if not cascade.empty():
                self._haar = cascade
                self.kind = "haar"
        except Exception:
            pass

    @property
    def available(self):
        return self.kind is not None

    def find_face(self, img_bgr):
        """Return (x, y, w, h) in pixels for the biggest face, or None."""
        h, w = img_bgr.shape[:2]
        if self.kind == "mediapipe":
            res = self._mp_det.process(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
            if not res.detections:
                return None
            best = max(
                res.detections,
                key=lambda d: d.location_data.relative_bounding_box.width
                * d.location_data.relative_bounding_box.height,
            )
            b = best.location_data.relative_bounding_box
            return int(b.xmin * w), int(b.ymin * h), int(b.width * w), int(b.height * h)
        if self.kind == "haar":
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            faces = self._haar.detectMultiScale(gray, 1.1, 4, minSize=(60, 60))
            if len(faces) == 0:
                return None
            x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
            return int(x), int(y), int(fw), int(fh)
        return None


def crop_with_margin(img, box, margin=0.2):
    h, w = img.shape[:2]
    x, y, bw, bh = box
    mx, my = int(bw * margin), int(bh * margin)
    x0, y0 = max(x - mx, 0), max(y - my, 0)
    x1, y1 = min(x + bw + mx, w), min(y + bh + my, h)
    if x1 <= x0 or y1 <= y0:
        return None
    return img[y0:y1, x0:x1]


def center_square(img):
    h, w = img.shape[:2]
    s = min(h, w)
    y0, x0 = (h - s) // 2, (w - s) // 2
    return img[y0:y0 + s, x0:x0 + s]


# ----------------------------------------------------------------- quality checks
def blur_score(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def brightness(img_bgr):
    return float(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).mean())


# ----------------------------------------------------------------- splitting
def choose_split(df, val_n, test_n, seed=42):
    """Pick which PEOPLE go to val and test.

    Tries every combination and prefers splits where val and test both contain all
    three classes, then the closest to the target 70/15/15 image proportions.
    """
    people = sorted(df["person_id"].unique())
    counts = df.groupby(["person_id", "label"]).size().unstack(fill_value=0)
    for c in CLASSES:
        if c not in counts.columns:
            counts[c] = 0
    total = len(df)
    best, best_key = None, None
    rng = random.Random(seed)
    for test_p in itertools.combinations(people, test_n):
        rest = [p for p in people if p not in test_p]
        for val_p in itertools.combinations(rest, val_n):
            train_p = [p for p in rest if p not in val_p]
            if not train_p:
                continue
            missing = 0
            for group in (val_p, test_p):
                have = counts.loc[list(group), CLASSES].sum()
                missing += int((have == 0).sum())
            n_val = int(counts.loc[list(val_p)].values.sum())
            n_test = int(counts.loc[list(test_p)].values.sum())
            err = abs(n_val / total - 0.15) + abs(n_test / total - 0.15)
            key = (missing, round(err, 4), rng.random())
            if best_key is None or key < best_key:
                best_key, best = key, (train_p, list(val_p), list(test_p))
    return best, best_key[0]


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Prepare the WakeWatch dataset")
    ap.add_argument("--input", required=True, help="folder containing p01..pNN")
    ap.add_argument("--output", default="data", help="where to write results")
    ap.add_argument("--blur-threshold", type=float, default=40.0,
                    help="reject face crops with Laplacian variance below this")
    ap.add_argument("--dup-distance", type=int, default=5,
                    help="phash distance at or below which images are duplicates")
    ap.add_argument("--keep-no-face", action="store_true",
                    help="if no face is found, keep the image (centre crop) instead of rejecting")
    ap.add_argument("--skip-face-crop", action="store_true",
                    help="do not detect faces at all; just centre-crop and resize")
    ap.add_argument("--val-people", type=int, default=None)
    ap.add_argument("--test-people", type=int, default=None)
    args = ap.parse_args()

    src, out = Path(args.input), Path(args.output)
    if not src.is_dir():
        sys.exit(f"Input folder not found: {src}")
    people = {}
    for p in sorted(src.iterdir()):
        digits = p.name[1:].replace("O", "0").replace("o", "0")
        if p.is_dir() and p.name[:1].lower() == "p" and digits.isdigit():
            people["p" + digits.zfill(2)] = p.name   # normalised id -> real folder name
    if not people:
        sys.exit("No person folders (p01, p02, ...) found inside the input folder.")

    cropper = FaceCropper()
    if not args.skip_face_crop and not cropper.available:
        sys.exit("No face detector available. Run: pip install mediapipe==0.10.14  "
                 "(or re-run with --skip-face-crop to just centre-crop).")
    print(f"Face detector: {'none (centre crop)' if args.skip_face_crop else cropper.kind}")

    if out.exists():
        for sub in ("clean_all", "_rejected", "reports"):
            shutil.rmtree(out / sub, ignore_errors=True)
    stage = out / "clean_all"
    rej = out / "_rejected"

    def reject(path, reason):
        dest = rej / reason
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest / f"{path.parent.parent.name}_{path.parent.name}_{path.name}")
        rejected.append((str(path), reason))

    rows, rejected = [], []
    total_in = 0
    for person, pfolder in people.items():
        seen = []   # duplicate check covers ALL classes of this person
        for cls in CLASSES:
            folder = next((d for d in (src / pfolder).iterdir()
                           if d.is_dir() and d.name.lower() == cls), None) \
                if (src / pfolder).is_dir() else None
            if folder is None:
                continue
            files = sorted(f for f in folder.iterdir() if f.suffix.lower() in IMG_EXT)
            for f in files:
                total_in += 1
                img = cv2.imread(str(f))
                if img is None or img.size == 0:
                    reject(f, "corrupted")
                    continue
                if min(img.shape[:2]) < 100:
                    reject(f, "too_small")
                    continue

                crop, used_face = None, False
                if not args.skip_face_crop:
                    box = cropper.find_face(img)
                    if box is not None:
                        crop = crop_with_margin(img, box)
                        used_face = crop is not None
                if crop is None:
                    if args.skip_face_crop or args.keep_no_face:
                        crop = center_square(img)
                    else:
                        reject(f, "no_face")
                        continue
                crop = cv2.resize(crop, (SIZE, SIZE), interpolation=cv2.INTER_AREA)

                if blur_score(crop) < args.blur_threshold:
                    reject(f, "blurry")
                    continue
                b = brightness(crop)
                if b < 30:
                    reject(f, "too_dark")
                    continue
                if b > 235:
                    reject(f, "overexposed")
                    continue

                h = imagehash.phash(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))
                match = next((s for s in seen if h - s["hash"] <= args.dup_distance), None)
                if match is not None:
                    if match["cls"] == cls and not match["conflict"]:
                        reject(f, "duplicate")
                        continue
                    # The same picture sits under two different labels: ambiguous.
                    # Remove BOTH copies from the dataset so you can decide by hand.
                    reject(f, "label_conflict")
                    if not match["conflict"]:
                        match["conflict"] = True
                        (stage / match["cls"] / match["name"]).unlink(missing_ok=True)
                        rows[:] = [r for r in rows if not (
                            r["filename"] == match["name"] and r["label"] == match["cls"])]
                        reject(match["src"], "label_conflict")
                    continue

                name = f"{person}_{cls}_{f.stem}.jpg"
                seen.append({"hash": h, "cls": cls, "name": name, "src": f, "conflict": False})
                dest = stage / cls
                dest.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(dest / name), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
                rows.append({"filename": name, "label": cls, "person_id": person,
                             "source": str(f), "face_found": used_face})

    if not rows:
        sys.exit("No images survived cleaning. Check the rejected folder and your settings.")
    df = pd.DataFrame(rows)

    # ---- split by person
    n_people = df["person_id"].nunique()
    test_n = args.test_people if args.test_people is not None else (2 if n_people >= 8 else 1)
    val_n = args.val_people if args.val_people is not None else 1
    if n_people < 3:
        sys.exit("Need at least 3 people to split by person.")
    (train_p, val_p, test_p), missing = choose_split(df, val_n, test_n)
    split_of = {**{p: "train" for p in train_p}, **{p: "val" for p in val_p},
                **{p: "test" for p in test_p}}
    df["split"] = df["person_id"].map(split_of)

    clean = out / "clean"
    shutil.rmtree(clean, ignore_errors=True)
    for _, r in df.iterrows():
        d = clean / r["split"] / r["label"]
        d.mkdir(parents=True, exist_ok=True)
        shutil.copy2(stage / r["label"] / r["filename"], d / r["filename"])

    # ---- verify no leakage
    leak = df.groupby("person_id")["split"].nunique().max()
    assert leak == 1, "A person appears in more than one split!"

    # ---- reports
    rep = out / "reports"
    rep.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "labels.csv", index=False)
    pd.DataFrame(rejected, columns=["file", "reason"]).to_csv(rep / "rejected_images.csv", index=False)

    lines = []
    lines.append(f"Images found:      {total_in}")
    lines.append(f"Images kept:       {len(df)}")
    lines.append(f"Images rejected:   {len(rejected)}")
    if rejected:
        for reason, n in pd.Series([r for _, r in rejected]).value_counts().items():
            lines.append(f"   - {reason}: {n}")
    lines.append("")
    lines.append("Kept per person and class:")
    lines.append(df.pivot_table(index="person_id", columns="label", values="filename",
                                aggfunc="count", fill_value=0).to_string())
    lines.append("")
    lines.append("Images per split and class:")
    lines.append(df.pivot_table(index="split", columns="label", values="filename",
                                aggfunc="count", fill_value=0).to_string())
    lines.append("")
    lines.append(f"train people: {', '.join(train_p)}")
    lines.append(f"val people:   {', '.join(val_p)}")
    lines.append(f"test people:  {', '.join(test_p)}")
    lines.append("Leakage check: PASSED (no person appears in more than one split)")

    warn = []
    per_class = df["label"].value_counts()
    if len(per_class) < 3:
        warn.append("A whole class has no images.")
    elif per_class.min() < 0.5 * per_class.max():
        warn.append(f"Class imbalance: {per_class.to_dict()} (smallest is under half of largest).")
    if missing:
        warn.append("Could not find a split where validation and test both contain all 3 classes.")
    one_class = [p for p, g in df.groupby("person_id") if g["label"].nunique() == 1]
    if one_class:
        warn.append(f"People with images in only one class: {', '.join(one_class)}")
    no_face = int((~df["face_found"]).sum())
    if no_face and not args.skip_face_crop:
        warn.append(f"{no_face} kept images had no detected face (centre-cropped).")
    lost = pd.Series([r for _, r in rejected]).eq("no_face").sum() if rejected else 0
    if lost:
        warn.append(f"{lost} images rejected for no face. Check _rejected/no_face for sleeping "
                    "photos; re-run with --keep-no-face to keep them.")
    if warn:
        lines.append("")
        lines.append("WARNINGS:")
        lines += [f"  ! {w}" for w in warn]

    summary = "\n".join(lines)
    (rep / "summary.txt").write_text(summary)
    print("\n" + summary)

    # ---- sample images for a visual check
    rnd = random.Random(1)
    for split in ("train", "val", "test"):
        sub = df[df["split"] == split]
        pick = sub.sample(min(20, len(sub)), random_state=1) if len(sub) else sub
        sd = rep / "samples" / split
        sd.mkdir(parents=True, exist_ok=True)
        for _, r in pick.iterrows():
            shutil.copy2(clean / split / r["label"] / r["filename"], sd / r["filename"])

    # ---- zip for Colab (cropped faces only)
    zpath = out / "clean.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for f in clean.rglob("*"):
            if f.is_file():
                z.write(f, f.relative_to(out))
    print(f"\nDone. Upload {zpath} to Google Colab.")
    print(f"Review the samples in {rep / 'samples'} and the rejects in {rej}.")


if __name__ == "__main__":
    main()
