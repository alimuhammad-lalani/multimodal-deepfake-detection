import torch
import torch.nn as nn
import torchvision.models as models

class DeepfakeDetectionModel(nn.Module):
    def __init__(self, num_frames=10):
        super().__init__()
        self.num_frames = num_frames
        res_v = models.resnet18(weights=None)
        res_a = models.resnet18(weights=None)
        self.video_features = nn.Sequential(*list(res_v.children())[:-1])
        self.audio_features = nn.Sequential(*list(res_a.children())[:-1])
        self.v_proj = nn.Linear(512, 256)
        self.a_proj = nn.Linear(512, 256)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=256, num_heads=8, batch_first=True
        )
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, audio_spec, video_frames):
        batch_size, num_f, c, h, w = video_frames.shape
        a_feat = self.audio_features(audio_spec).reshape(batch_size, -1)
        a_feat = self.a_proj(a_feat).unsqueeze(1)
        v_flat = video_frames.reshape(-1, c, h, w)
        v_feat = self.video_features(v_flat).reshape(batch_size, num_f, -1)
        v_feat = self.v_proj(v_feat)
        attn_output, _ = self.cross_attn(a_feat, v_feat, v_feat)
        return self.classifier(attn_output.squeeze(1))
