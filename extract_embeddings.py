import argparse
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModel

from config import (
    CHUNK_HOP_SEC,
    CHUNK_SEC,
    EMBEDDING_ROOT,
    FRAME_HOP_SEC,
    FRAME_SEC,
    MASTER_CSV,
    MERT_MODEL,
    SAM_ANNOTATED_SOURCES,
    SAM_TOLERANCE_SEC,
    SR,
)


def truthy(value):
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def safe_float(value, default=-1.0):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def load_sam_times(row):
    npz_path = row.get("npz_path", "")
    if not npz_path or not Path(npz_path).exists():
        return np.array([], dtype=np.float32)
    try:
        with np.load(npz_path, allow_pickle=True) as data:
            if "sam_times" in data:
                return np.asarray(data["sam_times"], dtype=np.float32)
            if "sama_times" in data:
                return np.asarray(data["sama_times"], dtype=np.float32)
    except Exception:
        pass
    return np.array([], dtype=np.float32)


def make_starts(duration, window_sec, hop_sec):
    if duration <= window_sec:
        return [0.0]
    starts = list(np.arange(0.0, max(0.0, duration - window_sec) + 1e-6, hop_sec))
    last = max(0.0, duration - window_sec)
    if not starts or abs(starts[-1] - last) > 1e-3:
        starts.append(last)
    return [float(x) for x in starts]


@torch.no_grad()
def embed_audio(model, audio, device):
    x = torch.from_numpy(audio).float().unsqueeze(0).to(device)
    hidden = model(x).last_hidden_state
    emb = hidden.mean(dim=1)
    return emb.squeeze(0).detach().cpu().numpy().astype(np.float16)


def extract_chunk_embeddings(model, wav_path, chunk_start, chunk_end, device):
    frame_starts = make_starts(chunk_end - chunk_start, FRAME_SEC, FRAME_HOP_SEC)
    embeddings, starts, ends = [], [], []

    for local_start in frame_starts:
        global_start = chunk_start + local_start
        global_end = min(global_start + FRAME_SEC, chunk_end)
        audio, _ = librosa.load(
            wav_path,
            sr=SR,
            mono=True,
            offset=global_start,
            duration=FRAME_SEC,
        )
        need = int(SR * FRAME_SEC)
        if len(audio) < need:
            audio = np.pad(audio, (0, need - len(audio)))

        embeddings.append(embed_audio(model, audio, device))
        starts.append(global_start)
        ends.append(global_end)

    return (
        np.stack(embeddings),
        np.asarray(starts, dtype=np.float32),
        np.asarray(ends, dtype=np.float32),
    )


def build_sam_target(frame_start, frame_end, sam_times):
    centers = 0.5 * (frame_start + frame_end)
    target = np.zeros(len(centers), dtype=np.float32)
    for t in sam_times:
        hit = np.abs(centers - float(t)) <= SAM_TOLERANCE_SEC
        target[hit] = 1.0
    return target


def process_row(row, model, device, overwrite=False):
    wav_path = str(row["wav_path"])
    if not Path(wav_path).exists():
        return 0

    split = str(row["split"]).strip()
    if split not in {"train", "val", "test"}:
        return 0

    source = str(row.get("source", "")).strip()
    recording_id = str(row.get("id", Path(wav_path).stem))
    out_dir = Path(EMBEDDING_ROOT) / split / recording_id
    out_dir.mkdir(parents=True, exist_ok=True)

    duration = librosa.get_duration(path=wav_path)
    chunk_starts = make_starts(duration, CHUNK_SEC, CHUNK_HOP_SEC)
    sam_times = load_sam_times(row)
    sam_annotated = source in SAM_ANNOTATED_SOURCES
    made = 0

    for chunk_idx, chunk_start in enumerate(chunk_starts):
        chunk_end = min(chunk_start + CHUNK_SEC, duration)
        out_file = out_dir / f"{recording_id}_chunk_{chunk_idx:04d}.npz"
        if out_file.exists() and not overwrite:
            continue

        emb, frame_start, frame_end = extract_chunk_embeddings(
            model, wav_path, chunk_start, chunk_end, device
        )
        sam_target = build_sam_target(frame_start, frame_end, sam_times)

        np.savez_compressed(
            out_file,
            embeddings=emb,
            frame_start=frame_start,
            frame_end=frame_end,
            sam_target=sam_target,
            sam_annotated=np.asarray(sam_annotated, dtype=np.bool_),
            taal=str(row["taal"]),
            tempo=np.asarray(safe_float(row.get("tempo", -1)), dtype=np.float32),
            scale=str(row.get("scale", "")),
            cycle_period=np.asarray(safe_float(row.get("cycle_period", -1)), dtype=np.float32),
            period_conf=np.asarray(safe_float(row.get("period_conf", 0), 0), dtype=np.float32),
            source=source,
            recording_id=recording_id,
            chunk_start=np.asarray(chunk_start, dtype=np.float32),
            chunk_end=np.asarray(chunk_end, dtype=np.float32),
            wav_path=wav_path,
        )
        made += 1

    return made


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=str(MASTER_CSV))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(str(MERT_MODEL), trust_remote_code=True).to(device)
    model.eval()

    df = pd.read_csv(args.csv)
    if "use" in df.columns:
        df = df[df["use"].map(truthy)]

    total = 0
    for _, row in tqdm(df.iterrows(), total=len(df)):
        try:
            total += process_row(row, model, device, overwrite=args.overwrite)
        except Exception as exc:
            print("FAILED", row.get("id", ""), exc)
    print("chunks written:", total)


if __name__ == "__main__":
    main()