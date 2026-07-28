import math

import torch
import torch.nn as nn

from config import D_MODEL, DROPOUT, EMBED_DIM, FF_DIM, NHEAD, NUM_LAYERS, SCALES, TAALS


USE_LOCAL_CONV = False          # set False to skip the conv block entirely.
LOCAL_CONV_KERNEL_SIZE = 9     # must be odd, so padding=(k-1)//2 keeps sequence length unchanged
LOCAL_CONV_GROUPS = 8         
LOCAL_CONV_PADDING = (LOCAL_CONV_KERNEL_SIZE - 1) // 2  


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class AttentionPool(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 1))

    def forward(self, x, mask):
        s = self.score(x).squeeze(-1)
        s = s.masked_fill(~mask, torch.finfo(s.dtype).min)
        w = torch.softmax(s, dim=1) * mask.float()
        w = w / (w.sum(dim=1, keepdim=True) + 1e-8)
        return (x * w.unsqueeze(-1)).sum(dim=1)


class TrackDSamModel(nn.Module):
    def __init__(self, n_taals=len(TAALS), n_scales=len(SCALES)):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(EMBED_DIM, D_MODEL), nn.LayerNorm(D_MODEL))

        self.use_local_conv = USE_LOCAL_CONV
        if self.use_local_conv:
            self.local = nn.Sequential(
                nn.Conv1d(
                    D_MODEL,
                    D_MODEL,
                    kernel_size=LOCAL_CONV_KERNEL_SIZE,
                    padding=LOCAL_CONV_PADDING,
                    groups=LOCAL_CONV_GROUPS,
                ),
                nn.GELU(),
                nn.Conv1d(D_MODEL, D_MODEL, kernel_size=1),
            )
        else:
            self.local = None

        self.cls_token = nn.Parameter(torch.zeros(1, 1, D_MODEL))
        self.pos = SinusoidalPositionalEncoding(D_MODEL)
        layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=NHEAD,
            dim_feedforward=FF_DIM,
            dropout=DROPOUT,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, NUM_LAYERS)
        self.pool = AttentionPool(D_MODEL)
        self.drop = nn.Dropout(0.2)

        self.taal_head = nn.Linear(D_MODEL, n_taals)
        self.scale_head = nn.Linear(D_MODEL, n_scales)
        self.tempo_head = nn.Sequential(nn.Linear(D_MODEL, 256), nn.GELU(), nn.Dropout(0.2), nn.Linear(256, 1))
        self.period_head = nn.Sequential(nn.Linear(D_MODEL, 256), nn.GELU(), nn.Dropout(0.2), nn.Linear(256, 1))
        self.sam_head = nn.Sequential(nn.Linear(D_MODEL, 256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256, 1))

    def forward(self, embeddings, mask):
        x = self.proj(embeddings)

        if self.use_local_conv:
            x = x + self.local(x.transpose(1, 2)).transpose(1, 2)

        batch = x.size(0)
        cls = self.cls_token.expand(batch, -1, -1)
        x = torch.cat([cls, x], dim=1)
        full_mask = torch.cat([torch.ones(batch, 1, device=mask.device, dtype=torch.bool), mask], dim=1)

        x = self.pos(x)
        x = self.encoder(x, src_key_padding_mask=~full_mask)

        tokens = x[:, 1:]
        pooled = self.pool(tokens, mask)
        feat = self.drop(0.0 * x[:, 0] + 1.0 * pooled)

        return {
            "taal": self.taal_head(feat),
            "scale": self.scale_head(feat),
            "tempo": self.tempo_head(feat).squeeze(-1),
            "period": self.period_head(feat).squeeze(-1),
            "sam": self.sam_head(tokens).squeeze(-1),
        }