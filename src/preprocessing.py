import argparse
import hashlib
import random
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from facenet_pytorch import MTCNN
from PIL import Image
from tqdm.auto import tqdm

AUDIO_LABELS = {
    "RealVideo-RealAudio": "real",
    "FakeVideo-RealAudio": "real",
    "RealVideo-FakeAudio": "fake",
    "FakeVideo-FakeAudio": "fake",
}


def build_metadata(dataset_root: str, train_split: float = 0.8, seed: int = 42) -> pd.DataFrame:
    """Scan FakeAVCeleb and create a speaker-disjoint train/validation split."""
    dataset_root = Path(dataset_root)
    rows = []

    for class_folder, label in AUDIO_LABELS.items():
        class_path = dataset_root / class_folder
        if not class_path.exists():
            continue

        for video_path in class_path.rglob("*.mp4"):
            speaker_id = next(
                (
                    part
                    for part in video_path.parts
                    if part.startswith("id") and part[2:].isdigit()
                ),
                video_path.parent.name,
            )

            rows.append(
                {
                    "path": video_path,
                    "speaker_id": speaker_id,
                    "label": label,
                }
            )

    df = pd.DataFrame(rows)

    if df.empty:
        raise RuntimeError(
            f"No MP4 files found under {dataset_root}. "
            "Check the FakeAVCeleb directory structure."
        )

    speakers = df["speaker_id"].drop_duplicates().tolist()
    rng = random.Random(seed)
    rng.shuffle(speakers)

    split_idx = int(len(speakers) * train_split)
    train_speakers = set(speakers[:split_idx])

    df["phase"] = df["speaker_id"].apply(
        lambda speaker: "train" if speaker in train_speakers else "validation"
    )

    return df


def extract_audio(
    metadata: pd.DataFrame,
    output_root: str,
    sample_rate: int = 16000,
    duration_seconds: int = 8,
) -> int:
    """Extract mono WAV audio with ffmpeg using deterministic filenames."""
    output_root = Path(output_root)

    for phase in ("train", "validation"):
        for label in ("real", "fake"):
            (output_root / phase / label).mkdir(parents=True, exist_ok=True)

    success = 0

    for _, row in tqdm(metadata.iterrows(), total=len(metadata), desc="Extracting audio"):
        source = Path(row["path"])
        uid = hashlib.md5(str(source).encode()).hexdigest()[:8]
        output_path = (
            output_root
            / row["phase"]
            / row["label"]
            / f"{row['speaker_id']}_{uid}.wav"
        )

        if output_path.exists() and output_path.stat().st_size > 0:
            success += 1
            continue

        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(source),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-af",
            f"apad=whole_dur={duration_seconds}",
            "-t",
            str(duration_seconds),
            str(output_path),
        ]

        result = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )

        if result.returncode == 0 and output_path.exists() and output_path.stat().st_size > 0:
            success += 1

    return success


def extract_faces(
    metadata: pd.DataFrame,
    output_root: str,
    num_frames: int = 10,
    image_size: int = 160,
    margin: int = 20,
) -> int:
    """Sample video frames and save MTCNN face crops as torch tensors."""
    output_root = Path(output_root)
    mtcnn = MTCNN(
        image_size=image_size,
        margin=margin,
        device="cpu",
        post_process=True,
    )

    for phase in ("train", "validation"):
        for label in ("real", "fake"):
            (output_root / phase / label).mkdir(parents=True, exist_ok=True)

    success = 0

    for _, row in tqdm(metadata.iterrows(), total=len(metadata), desc="Extracting faces"):
        source = Path(row["path"])
        uid = hashlib.md5(str(source).encode()).hexdigest()[:8]
        output_path = (
            output_root
            / row["phase"]
            / row["label"]
            / f"{row['speaker_id']}_{uid}.pth"
        )

        if output_path.exists():
            success += 1
            continue

        cap = cv2.VideoCapture(str(source))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames < num_frames:
            cap.release()
            continue

        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        frames = []

        for frame_idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ok, frame = cap.read()
            if not ok:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))

        cap.release()

        if len(frames) != num_frames:
            continue

        faces = mtcnn(frames)

        if faces is None or len(faces) != num_frames:
            continue

        torch.save(faces, output_path)
        success += 1

    return success


def check_speaker_overlap(metadata: pd.DataFrame) -> int:
    train_ids = set(metadata.loc[metadata["phase"] == "train", "speaker_id"])
    val_ids = set(metadata.loc[metadata["phase"] == "validation", "speaker_id"])
    return len(train_ids & val_ids)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--audio-out", required=True)
    parser.add_argument("--video-out", required=True)
    parser.add_argument("--train-split", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    metadata = build_metadata(
        dataset_root=args.dataset_root,
        train_split=args.train_split,
        seed=args.seed,
    )

    print(metadata.groupby(["phase", "label"]).size())
    print("Speaker overlap:", check_speaker_overlap(metadata))

    audio_success = extract_audio(metadata, args.audio_out)
    video_success = extract_faces(metadata, args.video_out)

    print(f"Audio files processed: {audio_success}/{len(metadata)}")
    print(f"Video tensors processed: {video_success}/{len(metadata)}")


if __name__ == "__main__":
    main()
