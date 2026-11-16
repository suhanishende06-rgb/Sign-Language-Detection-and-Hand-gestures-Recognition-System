import argparse
import csv
import hashlib
import json
import os
import re
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

import cv2
import mediapipe as mp
import numpy as np


BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "dataset" / "include_raw"
FEATURE_DIR = BASE_DIR / "dataset" / "include_dynamic"
RECORD_URL = "https://zenodo.org/api/records/4010759"
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
CATEGORY_NAMES = {
    "adjectives", "animals", "clothes", "colours", "days_and_time",
    "electronics", "greetings", "home", "jobs",
    "means_of_transportation", "people", "places", "pronouns",
    "seasons", "society",
}


def download_categories(categories):
    request = urllib.request.Request(
        RECORD_URL, headers={"User-Agent": "ISL-Sign-Recognition/1.0"}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        record = json.load(response)

    requested = {name.casefold().replace(" ", "_") for name in categories}
    files = [
        item for item in record.get("files", [])
        if item["key"].lower().endswith(".zip")
        and item["key"].rsplit("_", 1)[0].casefold() in requested
    ]
    if not files:
        raise RuntimeError(f"No INCLUDE ZIP files found for: {', '.join(categories)}")

    total_bytes = sum(item["size"] for item in files)
    print(f"Selected {len(files)} archives ({total_bytes / (1024 ** 3):.2f} GiB).")
    print("INCLUDE videos are large. Download only if you have enough disk space.")
    if input("Continue download? [y/N]: ").strip().lower() != "y":
        print("Download cancelled.")
        return

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for item in files:
        archive_path = RAW_DIR / item["key"]
        if archive_path.exists() and archive_path.stat().st_size == item["size"]:
            print(f"Already downloaded: {item['key']}")
            continue
        url = item.get("links", {}).get("self")
        if not url:
            raise RuntimeError(f"Zenodo did not provide a download URL for {item['key']}")

        print(f"Downloading {item['key']} ({item['size'] / (1024 ** 3):.2f} GiB)")
        request = urllib.request.Request(
            url, headers={"User-Agent": "ISL-Sign-Recognition/1.0"}
        )
        temporary_path = archive_path.with_suffix(archive_path.suffix + ".part")
        try:
            with urllib.request.urlopen(request, timeout=60) as response, temporary_path.open("wb") as target:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    target.write(chunk)
            if temporary_path.stat().st_size != item["size"]:
                raise RuntimeError(f"Incomplete download: {item['key']}")
            temporary_path.replace(archive_path)
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise


def safe_extract(archive_path, destination):
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            member_path = PurePosixPath(member.filename)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise ValueError(f"Unsafe path in archive {archive_path.name}: {member.filename}")
            target = destination.joinpath(*member_path.parts).resolve()
            if not target.is_relative_to(destination.resolve()):
                raise ValueError(f"Unsafe path in archive {archive_path.name}: {member.filename}")
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as source, target.open("wb") as output:
                    while True:
                        chunk = source.read(1024 * 1024)
                        if not chunk:
                            break
                        output.write(chunk)


def normalize_label(value):
    label = re.sub(r"[^A-Za-z0-9]+", "_", value.strip()).strip("_").upper()
    return label


def read_annotation_labels(root):
    labels = {}
    file_columns = {"video", "video_name", "filename", "file", "video_id", "id"}
    label_columns = {"gloss", "sign", "word", "label", "class"}
    for csv_path in root.rglob("*.csv"):
        try:
            with csv_path.open("r", encoding="utf-8-sig", newline="") as source:
                reader = csv.DictReader(source)
                if not reader.fieldnames:
                    continue
                columns = {name.strip().casefold(): name for name in reader.fieldnames}
                file_column = next(
                    (columns[name] for name in file_columns if name in columns), None
                )
                label_column = next(
                    (columns[name] for name in label_columns if name in columns), None
                )
                if not file_column or not label_column:
                    continue
                for row in reader:
                    file_value = (row.get(file_column) or "").strip()
                    label = normalize_label(row.get(label_column) or "")
                    if file_value and label:
                        key = Path(file_value.replace("\\", "/")).stem.casefold()
                        previous = labels.get(key)
                        if previous and previous != label:
                            raise ValueError(
                                f"Conflicting labels for video {file_value}: "
                                f"{previous} and {label}"
                            )
                        labels[key] = label
        except (OSError, UnicodeError, csv.Error) as error:
            print(f"Skipping unreadable annotation file {csv_path}: {error}")
    return labels


def infer_label(video_path, root, annotation_labels):
    label = annotation_labels.get(video_path.stem.casefold())
    if label:
        return label
    for parent in video_path.parents:
        if parent == root:
            break
        candidate = normalize_label(parent.name)
        if candidate and candidate.casefold() not in CATEGORY_NAMES and candidate.casefold() not in {
            "train", "test", "train_test_split", "include",
        }:
            return candidate
    return None


def extract_hand_features(results):
    if not results.hand_landmarks:
        return None
    features = np.zeros((2, 21, 3), dtype=np.float32)
    for index, landmarks in enumerate(results.hand_landmarks[:2]):
        slot = index
        if results.handedness and index < len(results.handedness) and results.handedness[index]:
            hand_name = results.handedness[index][0].category_name.lower()
            if hand_name == "left":
                slot = 0
            elif hand_name == "right":
                slot = 1
        points = np.array([[point.x, point.y, point.z] for point in landmarks], dtype=np.float32)
        points -= points[0].copy()
        scale = np.max(np.linalg.norm(points, axis=1))
        if scale > 1e-6:
            points /= scale
        features[slot] = points
    return features.reshape(126)


def resample_sequence(frames, sequence_length):
    if len(frames) < 5:
        return None
    indexes = np.linspace(0, len(frames) - 1, sequence_length).round().astype(int)
    return np.asarray(frames, dtype=np.float32)[indexes]


def extract_dataset(sequence_length, skip_frames):
    if not (RAW_DIR.exists() and any(RAW_DIR.rglob("*.zip"))):
        raise FileNotFoundError(
            f"No INCLUDE ZIP files found under {RAW_DIR}. Run this script with download first."
        )

    for archive_path in RAW_DIR.glob("*.zip"):
        destination = RAW_DIR / archive_path.stem
        if destination.exists() and any(destination.iterdir()):
            continue
        print(f"Extracting {archive_path.name}")
        safe_extract(archive_path, destination)

    videos = [
        path for path in RAW_DIR.rglob("*")
        if path.is_file() and path.suffix.casefold() in VIDEO_EXTENSIONS
    ]
    annotation_labels = read_annotation_labels(RAW_DIR)
    if not videos:
        raise RuntimeError(f"No video files found under {RAW_DIR} after extraction.")

    task_model = BASE_DIR / "hand_landmarker.task"
    if not task_model.is_file():
        raise FileNotFoundError(f"MediaPipe model not found: {task_model}")
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(task_model)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    with mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker:
        for index, video_path in enumerate(videos, start=1):
            label = infer_label(video_path, RAW_DIR, annotation_labels)
            if not label:
                print(
                    f"Skipped (no word label; need annotations or word folders): "
                    f"{video_path}"
                )
                continue
            output_dir = FEATURE_DIR / label
            relative_name = str(video_path.relative_to(RAW_DIR)).casefold().encode("utf-8")
            output_name = hashlib.sha1(relative_name).hexdigest() + ".npy"
            output_path = output_dir / output_name
            if output_path.exists():
                continue

            capture = cv2.VideoCapture(str(video_path))
            if not capture.isOpened():
                print(f"Skipped (cannot open video): {video_path}")
                continue
            frames = []
            frame_index = 0
            timestamp_ms = 0
            try:
                while True:
                    success, frame = capture.read()
                    if not success:
                        break
                    if frame_index % skip_frames == 0:
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                        timestamp_ms += 33 * skip_frames
                        results = landmarker.detect_for_video(image, timestamp_ms)
                        features = extract_hand_features(results)
                        if features is not None:
                            frames.append(features)
                    frame_index += 1
            finally:
                capture.release()

            sequence = resample_sequence(frames, sequence_length)
            if sequence is None:
                print(f"Skipped (too few hand frames): {video_path}")
                continue
            output_dir.mkdir(parents=True, exist_ok=True)
            np.save(output_path, sequence)
            print(f"[{index}/{len(videos)}] Prepared {label}: {video_path.name}")


def main():
    parser = argparse.ArgumentParser(
        description="Download and convert INCLUDE videos to the dynamic hand-sequence format."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    download_parser = subparsers.add_parser("download", help="Download chosen Zenodo categories")
    download_parser.add_argument(
        "--categories", nargs="+", default=["Greetings", "Colours"],
        help="INCLUDE categories to download (default: Greetings Colours)",
    )
    extract_parser = subparsers.add_parser("extract", help="Extract video landmarks to .npy sequences")
    extract_parser.add_argument("--sequence-length", type=int, default=30)
    extract_parser.add_argument("--skip-frames", type=int, default=2)
    args = parser.parse_args()

    if args.command == "download":
        download_categories(args.categories)
    else:
        if args.sequence_length < 2 or args.skip_frames < 1:
            parser.error("--sequence-length must be >= 2 and --skip-frames must be >= 1")
        extract_dataset(args.sequence_length, args.skip_frames)


if __name__ == "__main__":
    main()
