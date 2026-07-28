import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from config import (
    BATCH_SIZE,
    CHECKPOINT_DIR,
    EMBEDDING_ROOT,
    EPOCHS,
    GRAD_CLIP,
    LAMBDA_PERIOD,
    LAMBDA_SAM,
    LAMBDA_SCALE,
    LAMBDA_TAAL,
    LAMBDA_TEMPO,
    LR,
    NUM_WORKERS,
    PATIENCE,
    RESULT_DIR,
    SAM_POS_WEIGHT,
    SCALES,
    TAALS,
    WEIGHT_DECAY,
)
from dataset import SamFrameDataset, build_label_map, collate_fn
from model import TrackDSamModel


def masked_bce(logits, target, valid_mask, criterion):
    if not valid_mask.any():
        return logits.sum() * 0.0
    mask = valid_mask.unsqueeze(1).expand_as(target)
    return criterion(logits[mask], target[mask])


def print_dataset_summary(name, dataset):
    s = dataset.summary()
    print()
    print("=" * 72)
    print(f"{name.upper()} DATA")
    print("=" * 72)
    print(f"recordings          : {s['recordings']}")
    print(f"chunks              : {s['chunks']}")
    print(f"avg chunks/recording: {s['avg_chunks_per_recording']:.2f}")
    print(f"frames              : {s['total_frames']}")
    print(f"sam annotated chunks: {s['sam_annotated_chunks']}")
    print(f"sam positive frames : {s['sam_positive_frames']}")
    print(f"period real chunks  : {s['period_real_chunks']}")
    print(f"period derived      : {s['period_derived_chunks']}")
    print(f"tempo real chunks   : {s['tempo_real_chunks']}")
    print(f"tempo derived       : {s['tempo_derived_chunks']}")
    print(f"source chunks       : {s['by_source']}")
    print(f"taal chunks         : {s['by_taal']}")


def run_epoch(model, loader, device, optimizer=None, scaler=None):
    train = optimizer is not None
    model.train(train)
    ce = nn.CrossEntropyLoss(label_smoothing=0.08)
    reg = nn.SmoothL1Loss()
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(SAM_POS_WEIGHT, device=device))

    totals = {
        "loss": 0.0,
        "taal_ok": 0,
        "taal_n": 0,
        "scale_ok": 0,
        "scale_n": 0,
        "tempo_abs": 0.0,
        "tempo_n": 0,
        "tempo_abs_real": 0.0,
        "tempo_n_real": 0,
        "tempo_abs_derived": 0.0,
        "tempo_n_derived": 0,
        "period_abs": 0.0,
        "period_n": 0,
        "period_abs_real": 0.0,
        "period_n_real": 0,
        "period_abs_derived": 0.0,
        "period_n_derived": 0,
        "sam_tp": 0,
        "sam_fp": 0,
        "sam_fn": 0,
        "sam_tn": 0,
        "sam_frames": 0,
        "sam_positive_frames": 0,
        "sam_annotated_chunks": 0,
    }

    for batch in loader:
        emb = batch["embeddings"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        taal = batch["taal"].to(device)
        scale = batch["scale"].to(device)
        tempo = batch["tempo"].to(device)
        period = batch["cycle_period"].to(device)
        period_mask = batch["period_mask"].to(device)
        period_is_real = batch["period_is_real"].to(device)
        period_is_derived = batch["period_is_derived"].to(device)
        tempo_is_real = batch["tempo_is_real"].to(device)
        tempo_is_derived = batch["tempo_is_derived"].to(device)
        sam_target = batch["sam_target"].to(device)
        sam_annotated = batch["sam_annotated"].to(device)

        with torch.set_grad_enabled(train):
            with autocast(device_type="cuda", enabled=device.type == "cuda"):
                out = model(emb, mask)
                valid_scale = scale >= 0
                valid_tempo = tempo > 0

                loss = LAMBDA_TAAL * ce(out["taal"], taal)
                if valid_scale.any():
                    loss = loss + LAMBDA_SCALE * ce(out["scale"][valid_scale], scale[valid_scale])
                if valid_tempo.any():
                    loss = loss + LAMBDA_TEMPO * reg(torch.log1p(out["tempo"][valid_tempo].relu()), torch.log1p(tempo[valid_tempo]))
                if period_mask.any():
                    loss = loss + LAMBDA_PERIOD * reg(torch.log1p(out["period"][period_mask].relu()), torch.log1p(period[period_mask]))
                loss = loss + LAMBDA_SAM * masked_bce(out["sam"], sam_target, sam_annotated, bce)

            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()

        totals["loss"] += float(loss.detach().cpu())
        pred = out["taal"].argmax(1)
        totals["taal_ok"] += int((pred == taal).sum())
        totals["taal_n"] += int(taal.numel())
        if (scale >= 0).any():
            sp = out["scale"].argmax(1)
            totals["scale_ok"] += int((sp[scale >= 0] == scale[scale >= 0]).sum())
            totals["scale_n"] += int((scale >= 0).sum())
        if (tempo > 0).any():
            abs_err = (out["tempo"][tempo > 0] - tempo[tempo > 0]).abs()
            totals["tempo_abs"] += float(abs_err.sum())
            totals["tempo_n"] += int((tempo > 0).sum())
        if tempo_is_real.any():
            totals["tempo_abs_real"] += float((out["tempo"][tempo_is_real] - tempo[tempo_is_real]).abs().sum())
            totals["tempo_n_real"] += int(tempo_is_real.sum())
        if tempo_is_derived.any():
            totals["tempo_abs_derived"] += float((out["tempo"][tempo_is_derived] - tempo[tempo_is_derived]).abs().sum())
            totals["tempo_n_derived"] += int(tempo_is_derived.sum())
        if period_mask.any():
            totals["period_abs"] += float((out["period"][period_mask] - period[period_mask]).abs().sum())
            totals["period_n"] += int(period_mask.sum())
        if period_is_real.any():
            totals["period_abs_real"] += float((out["period"][period_is_real] - period[period_is_real]).abs().sum())
            totals["period_n_real"] += int(period_is_real.sum())
        if period_is_derived.any():
            totals["period_abs_derived"] += float((out["period"][period_is_derived] - period[period_is_derived]).abs().sum())
            totals["period_n_derived"] += int(period_is_derived.sum())
        if sam_annotated.any():
            valid_sam = sam_annotated.unsqueeze(1).expand_as(sam_target) & mask
            pred_sam = torch.sigmoid(out["sam"]) >= 0.5
            true_sam = sam_target >= 0.5
            totals["sam_tp"] += int((pred_sam & true_sam & valid_sam).sum())
            totals["sam_fp"] += int((pred_sam & ~true_sam & valid_sam).sum())
            totals["sam_fn"] += int((~pred_sam & true_sam & valid_sam).sum())
            totals["sam_tn"] += int((~pred_sam & ~true_sam & valid_sam).sum())
            totals["sam_frames"] += int(valid_sam.sum())
            totals["sam_positive_frames"] += int((true_sam & valid_sam).sum())
            totals["sam_annotated_chunks"] += int(sam_annotated.sum())

    sam_precision = totals["sam_tp"] / max(totals["sam_tp"] + totals["sam_fp"], 1)
    sam_recall = totals["sam_tp"] / max(totals["sam_tp"] + totals["sam_fn"], 1)
    sam_f1 = 2 * sam_precision * sam_recall / max(sam_precision + sam_recall, 1e-8)
    sam_acc = (totals["sam_tp"] + totals["sam_tn"]) / max(totals["sam_frames"], 1)

    return {
        "loss": totals["loss"] / max(len(loader), 1),
        "taal_acc": totals["taal_ok"] / max(totals["taal_n"], 1),
        "scale_acc": totals["scale_ok"] / max(totals["scale_n"], 1),
        "tempo_mae": totals["tempo_abs"] / max(totals["tempo_n"], 1),
        "tempo_mae_real": totals["tempo_abs_real"] / max(totals["tempo_n_real"], 1),
        "tempo_mae_derived": totals["tempo_abs_derived"] / max(totals["tempo_n_derived"], 1),
        "tempo_n_real": totals["tempo_n_real"],
        "tempo_n_derived": totals["tempo_n_derived"],
        "period_mae": totals["period_abs"] / max(totals["period_n"], 1),
        "period_mae_real": totals["period_abs_real"] / max(totals["period_n_real"], 1),
        "period_mae_derived": totals["period_abs_derived"] / max(totals["period_n_derived"], 1),
        "period_n_real": totals["period_n_real"],
        "period_n_derived": totals["period_n_derived"],
        "sam_precision": sam_precision,
        "sam_recall": sam_recall,
        "sam_f1": sam_f1,
        "sam_frame_acc": sam_acc,
        "sam_tp": totals["sam_tp"],
        "sam_fp": totals["sam_fp"],
        "sam_fn": totals["sam_fn"],
        "sam_tn": totals["sam_tn"],
        "sam_frames": totals["sam_frames"],
        "sam_positive_frames": totals["sam_positive_frames"],
        "sam_annotated_chunks": totals["sam_annotated_chunks"],
    }


def main():
    Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)
    Path(RESULT_DIR).mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    label_map = build_label_map(TAALS)
    scale_map = {s: i for i, s in enumerate(SCALES)}

    train_ds = SamFrameDataset(EMBEDDING_ROOT, "train", label_map, scale_map)
    val_ds = SamFrameDataset(EMBEDDING_ROOT, "val", label_map, scale_map)
    test_ds = SamFrameDataset(EMBEDDING_ROOT, "test", label_map, scale_map)
    print_dataset_summary("train", train_ds)
    print_dataset_summary("val", val_ds)
    print_dataset_summary("test", test_ds)
    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate_fn, drop_last=True)
    val_loader = DataLoader(val_ds, BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate_fn)

    model = TrackDSamModel(len(label_map), len(scale_map)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = GradScaler("cuda", enabled=device.type == "cuda")

    best_score = -1.0
    best_epoch = 0
    stale = 0
    history = []
    print()
    print("=" * 72)
    print("TRAINING START")
    print("=" * 72)
    print(f"epochs={EPOCHS} patience={PATIENCE} batch_size={BATCH_SIZE} lr={LR}")
    print("model init: fresh random TrackDSamModel head/encoder; no checkpoint loaded")
    print("best checkpoint criterion: 0.1*taal + 0.05*scale + 1*sam_f1 + 0.3/(1+period_mae)")
    for epoch in range(1, EPOCHS + 1):
        train_m = run_epoch(model, train_loader, device, optimizer, scaler)
        val_m = run_epoch(model, val_loader, device)
        scheduler.step()
        history.append({"epoch": epoch, **{f"train_{k}": v for k, v in train_m.items()}, **{f"val_{k}": v for k, v in val_m.items()}})

        score = (
            0.1 * val_m["taal_acc"]
            + 0.05 * val_m["scale_acc"]
            + 1 * val_m["sam_f1"]
            + 0.3 / (1 + val_m["period_mae"])
        )
        print(
            f"epoch {epoch:03d}/{EPOCHS} | "
            f"train_loss {train_m['loss']:.4f} | val_loss {val_m['loss']:.4f} | "
            f"taal {val_m['taal_acc']:.3f} | scale {val_m['scale_acc']:.3f} | "
            f"tempo_mae {val_m['tempo_mae']:.1f} (real {val_m['tempo_mae_real']:.1f} n={val_m['tempo_n_real']}, "
            f"derived {val_m['tempo_mae_derived']:.1f} n={val_m['tempo_n_derived']}) | "
            f"period_mae {val_m['period_mae']:.2f} (real {val_m['period_mae_real']:.2f} n={val_m['period_n_real']}, "
            f"derived {val_m['period_mae_derived']:.2f} n={val_m['period_n_derived']}) | "
            f"sam_f1 {val_m['sam_f1']:.3f} p {val_m['sam_precision']:.3f} r {val_m['sam_recall']:.3f} | "
            f"score {score:.4f}"
        )

        if score > best_score:
            best_score = score
            best_epoch = epoch
            stale = 0
            save_path = Path(CHECKPOINT_DIR) / "best.pt"
            torch.save({"model": model.state_dict(), "label_map": label_map, "scale_map": scale_map, "metrics": val_m, "epoch": epoch, "score": score}, save_path)
            print(f"  saved best checkpoint: {save_path} score={score:.4f}")
        else:
            stale += 1
            print(f"  no improvement: {stale}/{PATIENCE}")
            if stale >= PATIENCE:
                print(f"early stopping at epoch {epoch}; best epoch was {best_epoch}")
                break

    with open(Path(RESULT_DIR) / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"history saved: {Path(RESULT_DIR) / 'history.json'}")


if __name__ == "__main__":
    main()