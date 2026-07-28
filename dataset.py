from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from config import BEATS_PER_CYCLE, EMBED_DIM, SAM_ANNOTATED_SOURCES


CONF_THRESHOLD = 0.30


def build_label_map(items):
    return {name: i for i, name in enumerate(sorted(items))}


class SamFrameDataset(Dataset):
    def __init__(self, embedding_root, split, label_map, scale_map):
        self.label_map = label_map
        self.scale_map = scale_map
        root = Path(embedding_root) / split
        files = sorted(root.glob("*/*.npz")) + sorted(root.glob("*.npz"))
        self.files = []
        for path in files:
            try:
                with np.load(path, allow_pickle=True) as data:
                    if str(data["taal"]) in label_map:
                        self.files.append(path)
            except Exception:
                continue
        print(split, "chunks:", len(self.files))

    def summary(self):
        info = {
            "chunks": len(self.files),
            "recordings": 0,
            "avg_chunks_per_recording": 0.0,
            "sam_annotated_chunks": 0,
            "sam_positive_frames": 0,
            "total_frames": 0,
            "period_real_chunks": 0,
            "period_derived_chunks": 0,
            "tempo_real_chunks": 0,
            "tempo_derived_chunks": 0,
            "by_source": {},
            "by_taal": {},
        }
        recordings = set()
        for path in self.files:
            try:
                with np.load(path, allow_pickle=True) as data:
                    source = str(data.get("source", ""))
                    taal_name = str(data.get("taal", ""))
                    recording_id = str(data.get("recording_id", path.parent.name))
                    sam_annotated = bool(data.get("sam_annotated", source in SAM_ANNOTATED_SOURCES))
                    sam_target = data["sam_target"]
                    n_frames = int(len(sam_target))

                    tempo = float(data.get("tempo", -1))
                    cycle_period = float(data.get("cycle_period", -1))
                    period_conf = float(data.get("period_conf", 0))
                    period_is_real = cycle_period > 0 and period_conf >= CONF_THRESHOLD
                    tempo_is_real = tempo > 0
                    period_is_derived = (not period_is_real) and tempo_is_real and taal_name in BEATS_PER_CYCLE
                    tempo_is_derived = (not tempo_is_real) and period_is_real and taal_name in BEATS_PER_CYCLE

                    recordings.add(recording_id)
                    info["total_frames"] += n_frames
                    info["sam_positive_frames"] += int(np.asarray(sam_target).sum())
                    if sam_annotated:
                        info["sam_annotated_chunks"] += 1
                    if period_is_real:
                        info["period_real_chunks"] += 1
                    if period_is_derived:
                        info["period_derived_chunks"] += 1
                    if tempo_is_real:
                        info["tempo_real_chunks"] += 1
                    if tempo_is_derived:
                        info["tempo_derived_chunks"] += 1
                    info["by_source"][source] = info["by_source"].get(source, 0) + 1
                    info["by_taal"][taal_name] = info["by_taal"].get(taal_name, 0) + 1
            except Exception:
                continue
        info["recordings"] = len(recordings)
        info["avg_chunks_per_recording"] = info["chunks"] / max(info["recordings"], 1)
        return info

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        with np.load(path, allow_pickle=True) as data:
            emb = data["embeddings"].astype(np.float32)
            if emb.ndim != 2 or emb.shape[1] != EMBED_DIM:
                raise ValueError(f"{path}: bad embedding shape {emb.shape}")

            source = str(data["source"])
            taal_name = str(data["taal"])
            sam_annotated = bool(data.get("sam_annotated", source in SAM_ANNOTATED_SOURCES))
            scale = str(data.get("scale", ""))
            tempo = float(data.get("tempo", -1))
            cycle_period = float(data.get("cycle_period", -1))
            period_conf = float(data.get("period_conf", 0))

            period_is_real = cycle_period > 0 and period_conf >= CONF_THRESHOLD
            period_is_derived = False
            tempo_is_real = tempo > 0
            tempo_is_derived = False

            # period missing/unreliable but tempo is real -> derive period from tempo
            if not period_is_real and tempo_is_real and taal_name in BEATS_PER_CYCLE:
                cycle_period = BEATS_PER_CYCLE[taal_name] * 60.0 / tempo
                period_is_derived = True

            # tempo missing but period is REAL (not derived, avoid circularity) -> derive tempo from period
            if not tempo_is_real and period_is_real and taal_name in BEATS_PER_CYCLE:
                tempo = BEATS_PER_CYCLE[taal_name] * 60.0 / cycle_period
                tempo_is_derived = True

            return {
                "embeddings": torch.from_numpy(emb),
                "taal": torch.tensor(self.label_map[taal_name], dtype=torch.long),
                "taal_name": taal_name,
                "scale": torch.tensor(self.scale_map.get(scale, -1), dtype=torch.long),
                "scale_name": scale,
                "tempo": torch.tensor(tempo, dtype=torch.float32),
                "cycle_period": torch.tensor(cycle_period, dtype=torch.float32),
                "period_mask": torch.tensor(period_is_real or period_is_derived),
                "period_is_real": torch.tensor(period_is_real),
                "period_is_derived": torch.tensor(period_is_derived),
                "tempo_mask": torch.tensor(tempo_is_real or tempo_is_derived),
                "tempo_is_real": torch.tensor(tempo_is_real),
                "tempo_is_derived": torch.tensor(tempo_is_derived),
                "sam_target": torch.from_numpy(data["sam_target"].astype(np.float32)),
                "sam_annotated": torch.tensor(sam_annotated, dtype=torch.bool),
                "frame_start": torch.from_numpy(data["frame_start"].astype(np.float32)),
                "frame_end": torch.from_numpy(data["frame_end"].astype(np.float32)),
                "source": source,
                "recording_id": str(data.get("recording_id", "")),
                "file": str(path),
            }


def collate_fn(batch):
    batch_size = len(batch)
    tmax = max(item["embeddings"].shape[0] for item in batch)
    dim = batch[0]["embeddings"].shape[1]

    emb = torch.zeros(batch_size, tmax, dim)
    mask = torch.zeros(batch_size, tmax, dtype=torch.bool)
    sam = torch.zeros(batch_size, tmax)
    frame_start = torch.zeros(batch_size, tmax)
    frame_end = torch.zeros(batch_size, tmax)

    for i, item in enumerate(batch):
        n = item["embeddings"].shape[0]
        emb[i, :n] = item["embeddings"]
        mask[i, :n] = True
        sam[i, :n] = item["sam_target"]
        frame_start[i, :n] = item["frame_start"]
        frame_end[i, :n] = item["frame_end"]

    return {
        "embeddings": emb,
        "mask": mask,
        "sam_target": sam,
        "frame_start": frame_start,
        "frame_end": frame_end,
        "taal": torch.stack([x["taal"] for x in batch]),
        "scale": torch.stack([x["scale"] for x in batch]),
        "tempo": torch.stack([x["tempo"] for x in batch]),
        "cycle_period": torch.stack([x["cycle_period"] for x in batch]),
        "period_mask": torch.stack([x["period_mask"] for x in batch]),
        "period_is_real": torch.stack([x["period_is_real"] for x in batch]),
        "period_is_derived": torch.stack([x["period_is_derived"] for x in batch]),
        "tempo_mask": torch.stack([x["tempo_mask"] for x in batch]),
        "tempo_is_real": torch.stack([x["tempo_is_real"] for x in batch]),
        "tempo_is_derived": torch.stack([x["tempo_is_derived"] for x in batch]),
        "sam_annotated": torch.stack([x["sam_annotated"] for x in batch]),
        "taal_name": [x["taal_name"] for x in batch],
        "scale_name": [x["scale_name"] for x in batch],
        "source": [x["source"] for x in batch],
        "recording_id": [x["recording_id"] for x in batch],
        "file": [x["file"] for x in batch],
    }