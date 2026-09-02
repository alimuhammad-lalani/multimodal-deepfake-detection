from pathlib import Path
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset

class DeepfakeDataset(Dataset):
    def __init__(self, audio_root, video_root, phase="train", sample_rate=16000):
        self.audio_root = Path(audio_root) / phase
        self.video_root = Path(video_root) / phase
        self.sample_rate = sample_rate
        self.target_len = 32000
        self.audio_transform = T.MelSpectrogram(
            sample_rate=sample_rate, n_mels=128, n_fft=1024, hop_length=512
        )
        self.samples = []
        for label in ("real", "fake"):
            v_folder = self.video_root / label
            if not v_folder.exists():
                continue
            for v_path in v_folder.glob("*.pth"):
                uid = v_path.stem
                a_path = self.audio_root / label / f"{uid}.wav"
                if a_path.exists():
                    y = 1.0 if label == "fake" else 0.0
                    self.samples.append((a_path, v_path, y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        a_path, v_path, label = self.samples[idx]
        waveform, sr = torchaudio.load(a_path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, sr, self.sample_rate
            )
        if waveform.shape[1] > self.target_len:
            waveform = waveform[:, :self.target_len]
        else:
            pad = self.target_len - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad))

        spec = self.audio_transform(waveform)
        spec = torch.log(spec + 1e-9).repeat(3, 1, 1)

        video_data = torch.load(v_path, map_location="cpu")
        video_tensor = (
            torch.stack(video_data) if isinstance(video_data, list) else video_data
        ).float()

        return spec.float(), video_tensor, torch.tensor(label, dtype=torch.float32)
