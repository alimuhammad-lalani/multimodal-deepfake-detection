# Running the Pipeline

## 1. Preprocess FakeAVCeleb

```bash
python src/preprocessing.py \
  --dataset-root /path/to/FakeAVCeleb_v1.2 \
  --audio-out /path/to/processed_audio \
  --video-out /path/to/processed_video
```

The preprocessing script:
- builds an 80/20 speaker-disjoint split,
- extracts mono 16 kHz audio,
- samples 10 frames per video,
- detects/crops faces with MTCNN,
- and saves paired deterministic audio/video filenames.

## 2. Train

```bash
python src/train.py \
  --audio-root /path/to/processed_audio \
  --video-root /path/to/processed_video \
  --output-dir artifacts
```

Model checkpoints and metric history are written to `artifacts/`, which is gitignored.
