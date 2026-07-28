import argparse
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel

from config import (
    BEATS_PER_CYCLE,
    CHECKPOINT_DIR,
    CHUNK_HOP_SEC,
    CHUNK_SEC,
    FRAME_HOP_SEC,
    FRAME_SEC,
    MERT_MODEL,
    SR,
)
from model import TrackDSamModel



def make_starts(duration, window_sec, hop_sec):
    if duration <= window_sec:
        return [0.0]
    starts = list(np.arange(0.0, max(0.0, duration - window_sec) + 1e-6, hop_sec))
    last = max(0.0, duration - window_sec)
    if not starts or abs(starts[-1] - last) > 1e-3:
        starts.append(last)
    return [float(x) for x in starts]


@torch.no_grad()
def embed_audio(mert_model, audio, device):
    x = torch.from_numpy(audio).float().unsqueeze(0).to(device)
    hidden = mert_model(x).last_hidden_state
    emb = hidden.mean(dim=1)
    return emb.squeeze(0).detach().cpu().numpy().astype(np.float32)


def extract_chunk_embeddings(mert_model, wav_path, chunk_start, chunk_end, device):
    frame_starts = make_starts(chunk_end - chunk_start, FRAME_SEC, FRAME_HOP_SEC)
    embeddings, starts, ends = [], [], []

    for local_start in frame_starts:
        global_start = chunk_start + local_start
        global_end = min(global_start + FRAME_SEC, chunk_end)
        audio, _ = librosa.load(
            wav_path, sr=SR, mono=True, offset=global_start, duration=FRAME_SEC
        )
        need = int(SR * FRAME_SEC)
        if len(audio) < need:
            audio = np.pad(audio, (0, need - len(audio)))

        embeddings.append(embed_audio(mert_model, audio, device))
        starts.append(global_start)
        ends.append(global_end)

    return (
        np.stack(embeddings),
        np.asarray(starts, dtype=np.float32),
        np.asarray(ends, dtype=np.float32),
    )


def grouped_event_times(times, scores, threshold=0.5, merge_gap=1.0):
    """Same merge logic as eval.py's grouped_event_times, standalone here
    so infer.py has no dependency on eval.py."""
    times = np.asarray(times, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    keep = scores >= threshold
    times, scores = times[keep], scores[keep]
    if len(times) == 0:
        return []

    order = np.argsort(times)
    times, scores = times[order], scores[order]
    groups = []
    current = [(float(times[0]), float(scores[0]))]
    for t, s in zip(times[1:], scores[1:]):
        if float(t) - current[-1][0] <= merge_gap:
            current.append((float(t), float(s)))
        else:
            groups.append(current)
            current = [(float(t), float(s))]
    groups.append(current)

    return [max(group, key=lambda x: x[1])[0] for group in groups]


def run_inference(wav_path, ckpt_path, sam_threshold=0.5, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    label_map = ckpt["label_map"]
    scale_map = ckpt["scale_map"]
    inv_taal = {v: k for k, v in label_map.items()}
    inv_scale = {v: k for k, v in scale_map.items()}

    model = TrackDSamModel(len(label_map), len(scale_map)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Model loaded (trained to epoch {ckpt.get('epoch')}, val score {ckpt.get('score'):.4f})")

    print(f"Loading MERT feature extractor: {MERT_MODEL}")
    mert_model = AutoModel.from_pretrained(str(MERT_MODEL), trust_remote_code=True).to(device)
    mert_model.eval()

    print(f"\nProcessing audio: {wav_path}")
    duration = librosa.get_duration(path=wav_path)
    chunk_starts = make_starts(duration, CHUNK_SEC, CHUNK_HOP_SEC)
    print(f"Duration: {duration:.1f}s -> {len(chunk_starts)} chunk(s) of {CHUNK_SEC:.0f}s (hop {CHUNK_HOP_SEC:.0f}s)")

    all_taal_probs = []
    all_scale_probs = []
    all_tempo_preds = []
    all_period_preds = []
    all_sam_times = []
    all_sam_probs = []

    for idx, chunk_start in enumerate(chunk_starts):
        chunk_end = min(chunk_start + CHUNK_SEC, duration)
        emb, frame_start, frame_end = extract_chunk_embeddings(
            mert_model, wav_path, chunk_start, chunk_end, device
        )
        centers = 0.5 * (frame_start + frame_end)

        emb_t = torch.from_numpy(emb).unsqueeze(0).to(device)  # [1, T, 768]
        mask_t = torch.ones(1, emb_t.size(1), dtype=torch.bool, device=device)

        with torch.no_grad():
            out = model(emb_t, mask_t)

        taal_probs = F.softmax(out["taal"], dim=1).cpu().numpy()[0]
        scale_probs = F.softmax(out["scale"], dim=1).cpu().numpy()[0]
        tempo_pred = float(out["tempo"].cpu().item())
        period_pred = float(out["period"].cpu().item())
        sam_probs = torch.sigmoid(out["sam"]).cpu().numpy()[0]

        all_taal_probs.append(taal_probs)
        all_scale_probs.append(scale_probs)
        all_tempo_preds.append(tempo_pred)
        all_period_preds.append(period_pred)
        all_sam_times.extend(centers.tolist())
        all_sam_probs.extend(sam_probs.tolist())

        print(f"  chunk {idx+1}/{len(chunk_starts)} [{chunk_start:6.1f}s - {chunk_end:6.1f}s] processed")

    # ---- aggregate across chunks (recording-level, matching eval.py) ----
    taal_probs_avg = np.mean(all_taal_probs, axis=0)
    scale_probs_avg = np.mean(all_scale_probs, axis=0)
    taal_pred = inv_taal[int(np.argmax(taal_probs_avg))]
    taal_conf = float(np.max(taal_probs_avg))
    scale_pred = inv_scale[int(np.argmax(scale_probs_avg))]
    scale_conf = float(np.max(scale_probs_avg))
    tempo_pred = float(np.median(all_tempo_preds))
    period_pred = float(np.median(all_period_preds))

    sam_events = grouped_event_times(all_sam_times, all_sam_probs, threshold=sam_threshold)
    sam_events = sorted(sam_events)

    # cross-check: does tempo/period agree with beats-per-cycle formula?
    beats = BEATS_PER_CYCLE.get(taal_pred)
    formula_period = (beats * 60.0 / tempo_pred) if (beats and tempo_pred > 0) else None
    formula_tempo = (beats * 60.0 / period_pred) if (beats and period_pred > 0) else None

    print()
    print("=" * 72)
    print("PREDICTION SUMMARY")
    print("=" * 72)
    print(f"File               : {wav_path}")
    print(f"Duration           : {duration:.1f}s")
    print(f"Chunks processed   : {len(chunk_starts)}")
    print()
    print(f"Taal               : {taal_pred}  (confidence {taal_conf*100:.1f}%)")
    print(f"Scale              : {scale_pred}  (confidence {scale_conf*100:.1f}%)")
    print(f"Tempo              : {tempo_pred:.1f} BPM")
    print(f"Cycle period       : {period_pred:.2f} sec")
    if formula_period is not None:
        print(f"  (cross-check: beats_per_cycle({beats}) x 60 / tempo = {formula_period:.2f}s -- "
              f"{'consistent' if abs(formula_period - period_pred) < 1.0 else 'INCONSISTENT'} with period prediction)")
    print()
    print(f"Sam events detected: {len(sam_events)}  (threshold={sam_threshold})")
    if sam_events:
        preview = ", ".join(f"{t:.2f}s" for t in sam_events[:15])
        more = f"  ... (+{len(sam_events)-15} more)" if len(sam_events) > 15 else ""
        print(f"  Timestamps: {preview}{more}")
        if len(sam_events) >= 2:
            gaps = np.diff(sam_events)
            print(f"  Inter-sam gap: mean={np.mean(gaps):.2f}s, median={np.median(gaps):.2f}s "
                  f"(model's own predicted period: {period_pred:.2f}s)")
    else:
        print("  (none above threshold)")
    print("=" * 72)

    return {
        "duration_sec": duration,
        "n_chunks": len(chunk_starts),
        "taal": taal_pred,
        "taal_confidence": taal_conf,
        "scale": scale_pred,
        "scale_confidence": scale_conf,
        "tempo_bpm": tempo_pred,
        "cycle_period_sec": period_pred,
        "sam_event_times": sam_events,
    }


def main():
    parser = argparse.ArgumentParser(description="Run Track D inference on a single audio file")
    parser.add_argument("wav_path", help="Path to input audio file (wav/mp3/etc.)")
    parser.add_argument("--ckpt", default=str(Path(CHECKPOINT_DIR) / "best.pt"))
    parser.add_argument("--sam-threshold", type=float, default=0.5)
    args = parser.parse_args()

    if not Path(args.wav_path).exists():
        raise FileNotFoundError(f"Audio file not found: {args.wav_path}")

    run_inference(args.wav_path, args.ckpt, sam_threshold=args.sam_threshold)


if __name__ == "__main__":
    main()