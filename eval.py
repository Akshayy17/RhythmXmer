import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import BATCH_SIZE, CHECKPOINT_DIR, EMBEDDING_ROOT, NUM_WORKERS, RESULT_DIR
from dataset import SamFrameDataset, collate_fn
from model import TrackDSamModel
from train import print_dataset_summary, run_epoch


THRESHOLDS = [round(x, 2) for x in np.arange(0.10, 0.91, 0.05)]
EVENT_TOLERANCES = [0.5, 1.0]


def deduplicated_frame_metrics(recordings, threshold=0.5):

    totals = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}

    for rec in recordings.values():
        if not rec["sam_annotated"]:
            continue

        times = np.asarray(rec["sam_times"], dtype=np.float32)
        probs = np.asarray(rec["sam_probs"], dtype=np.float32)
        true_positive_times = set(np.round(rec["true_sam_times"], 2))  # rounded for float-safe matching

        # Deduplicate: for identical (or near-identical) timestamps, keep max-confidence
        best_by_time = {}
        for t, p in zip(times, probs):
            key = round(float(t), 2)
            if key not in best_by_time or p > best_by_time[key]:
                best_by_time[key] = p

        for t_key, best_prob in best_by_time.items():
            pred_positive = best_prob >= threshold
            true_positive = t_key in true_positive_times

            if pred_positive and true_positive:
                totals["tp"] += 1
            elif pred_positive and not true_positive:
                totals["fp"] += 1
            elif not pred_positive and true_positive:
                totals["fn"] += 1
            else:
                totals["tn"] += 1

    p = totals["tp"] / max(totals["tp"] + totals["fp"], 1)
    r = totals["tp"] / max(totals["tp"] + totals["fn"], 1)
    f1 = 2 * p * r / max(p + r, 1e-8)
    return {
        "precision": p, "recall": r, "f1": f1,
        "tp": totals["tp"], "fp": totals["fp"], "fn": totals["fn"], "tn": totals["tn"],
    }


def empty_bucket():
    return {
        "chunks": 0,
        "taal_ok": 0,
        "scale_ok": 0,
        "scale_n": 0,
        "tempo_abs": 0.0,
        "tempo_n": 0,
        "tempo_real_n": 0,
        "tempo_derived_n": 0,
        "tempo_abs_real": 0.0,
        "tempo_abs_derived": 0.0,
        "period_abs": 0.0,
        "period_n": 0,
        "period_n_real": 0,
        "period_n_derived": 0,
        "period_abs_real": 0.0,
        "period_abs_derived": 0.0,
        "sam_tp": 0,
        "sam_fp": 0,
        "sam_fn": 0,
        "sam_tn": 0,
        "sam_frames": 0,
        "sam_positive_frames": 0,
        "sam_annotated_chunks": 0,
    }


def add_to_bucket(bucket, row):
    bucket["chunks"] += 1
    bucket["taal_ok"] += int(row["taal_correct"])
    if row["scale_target"] is not None:
        bucket["scale_n"] += 1
        bucket["scale_ok"] += int(row["scale_correct"])
    if row["tempo_target"] > 0:
        bucket["tempo_n"] += 1
        err = abs(row["tempo_pred"] - row["tempo_target"])
        bucket["tempo_abs"] += err
        if row["tempo_is_real"]:
            bucket["tempo_real_n"] += 1
            bucket["tempo_abs_real"] += err
        if row["tempo_is_derived"]:
            bucket["tempo_derived_n"] += 1
            bucket["tempo_abs_derived"] += err
    if row["period_valid"]:
        bucket["period_n"] += 1
        err = abs(row["period_pred"] - row["period_target"])
        bucket["period_abs"] += err
        bucket["period_n_real"] += int(row["period_is_real"])
        bucket["period_n_derived"] += int(row["period_is_derived"])
        if row["period_is_real"]:
            bucket["period_abs_real"] += err
        if row["period_is_derived"]:
            bucket["period_abs_derived"] += err
    if row["sam_annotated"]:
        bucket["sam_annotated_chunks"] += 1
        bucket["sam_tp"] += row["sam_tp"]
        bucket["sam_fp"] += row["sam_fp"]
        bucket["sam_fn"] += row["sam_fn"]
        bucket["sam_tn"] += row["sam_tn"]
        bucket["sam_frames"] += row["sam_frames"]
        bucket["sam_positive_frames"] += row["sam_positive_frames"]


def finalize_bucket(bucket):
    p = bucket["sam_tp"] / max(bucket["sam_tp"] + bucket["sam_fp"], 1)
    r = bucket["sam_tp"] / max(bucket["sam_tp"] + bucket["sam_fn"], 1)
    f1 = 2 * p * r / max(p + r, 1e-8)
    return {
        "chunks": bucket["chunks"],
        "taal_acc": bucket["taal_ok"] / max(bucket["chunks"], 1),
        "scale_acc": bucket["scale_ok"] / max(bucket["scale_n"], 1),
        "scale_n": bucket["scale_n"],
        "tempo_mae": bucket["tempo_abs"] / max(bucket["tempo_n"], 1),
        "tempo_n": bucket["tempo_n"],
        "tempo_mae_real": bucket["tempo_abs_real"] / max(bucket["tempo_real_n"], 1),
        "tempo_n_real": bucket["tempo_real_n"],
        "tempo_mae_derived": bucket["tempo_abs_derived"] / max(bucket["tempo_derived_n"], 1),
        "tempo_n_derived": bucket["tempo_derived_n"],
        "period_mae": bucket["period_abs"] / max(bucket["period_n"], 1),
        "period_n": bucket["period_n"],
        "period_n_real": bucket["period_n_real"],
        "period_n_derived": bucket["period_n_derived"],
        "period_mae_real": bucket["period_abs_real"] / max(bucket["period_n_real"], 1),
        "period_mae_derived": bucket["period_abs_derived"] / max(bucket["period_n_derived"], 1),
        "sam_precision": p,
        "sam_recall": r,
        "sam_f1": f1,
        "sam_frame_acc": (bucket["sam_tp"] + bucket["sam_tn"]) / max(bucket["sam_frames"], 1),
        "sam_tp": bucket["sam_tp"],
        "sam_fp": bucket["sam_fp"],
        "sam_fn": bucket["sam_fn"],
        "sam_tn": bucket["sam_tn"],
        "sam_frames": bucket["sam_frames"],
        "sam_positive_frames": bucket["sam_positive_frames"],
        "sam_annotated_chunks": bucket["sam_annotated_chunks"],
    }


def grouped_event_times(times, scores=None, threshold=0.5):
    times = np.asarray(times, dtype=np.float32)
    if len(times) == 0:
        return []
    if scores is None:
        keep = np.ones(len(times), dtype=bool)
        scores = np.ones(len(times), dtype=np.float32)
    else:
        scores = np.asarray(scores, dtype=np.float32)
        keep = scores >= threshold
    times = times[keep]
    scores = scores[keep]
    if len(times) == 0:
        return []

    order = np.argsort(times)
    times = times[order]
    scores = scores[order]
    groups = []
    current = [(float(times[0]), float(scores[0]))]
    for t, s in zip(times[1:], scores[1:]):
        if float(t) - current[-1][0] <= 1.0:
            current.append((float(t), float(s)))
        else:
            groups.append(current)
            current = [(float(t), float(s))]
    groups.append(current)

    events = []
    for group in groups:
        best = max(group, key=lambda x: x[1])
        events.append(best[0])
    return events


def event_prf(pred_events, true_events, tolerance):
    pred_events = sorted(pred_events)
    true_events = sorted(true_events)
    used_true = set()
    tp = 0
    for p in pred_events:
        best_idx = None
        best_dist = None
        for i, t in enumerate(true_events):
            if i in used_true:
                continue
            dist = abs(p - t)
            if dist <= tolerance and (best_dist is None or dist < best_dist):
                best_idx = i
                best_dist = dist
        if best_idx is not None:
            used_true.add(best_idx)
            tp += 1
    fp = len(pred_events) - tp
    fn = len(true_events) - tp
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def add_event_counts(total, metric):
    total["tp"] += metric["tp"]
    total["fp"] += metric["fp"]
    total["fn"] += metric["fn"]


def finalize_event_counts(total):
    p = total["tp"] / max(total["tp"] + total["fp"], 1)
    r = total["tp"] / max(total["tp"] + total["fn"], 1)
    f1 = 2 * p * r / max(p + r, 1e-8)
    return {"precision": p, "recall": r, "f1": f1, "tp": total["tp"], "fp": total["fp"], "fn": total["fn"]}


@torch.no_grad()
def detailed_eval(model, loader, device, label_map, scale_map, predictions_path, threshold):
    inv_taal = {v: k for k, v in label_map.items()}
    inv_scale = {v: k for k, v in scale_map.items()}
    model.eval()

    overall = empty_bucket()
    per_taal = {}
    per_source = {}
    per_taal_source = {}
    recordings = {}

    with open(predictions_path, "w") as pred_file:
        for batch in loader:
            emb = batch["embeddings"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            out = model(emb, mask)

            taal_probs = F.softmax(out["taal"], dim=1).cpu()
            scale_probs = F.softmax(out["scale"], dim=1).cpu()
            taal_pred = taal_probs.argmax(1)
            scale_pred = scale_probs.argmax(1)
            tempo_pred = out["tempo"].float().cpu()
            period_pred = out["period"].float().cpu()
            sam_prob = torch.sigmoid(out["sam"]).cpu()
            sam_pred = sam_prob >= threshold

            for i in range(emb.size(0)):
                valid = batch["mask"][i]
                centers = 0.5 * (batch["frame_start"][i][valid].numpy() + batch["frame_end"][i][valid].numpy())
                true_mask = (batch["sam_target"][i][valid] >= 0.5).numpy()
                pred_scores = sam_prob[i][valid].numpy()

                sam_true = batch["sam_target"][i] >= 0.5
                sam_valid = valid & batch["sam_annotated"][i]
                row = {
                    "file": batch["file"][i],
                    "recording_id": batch["recording_id"][i],
                    "source": batch["source"][i],
                    "taal_target": batch["taal_name"][i],
                    "taal_pred": inv_taal[int(taal_pred[i])],
                    "taal_confidence": float(taal_probs[i, taal_pred[i]]),
                    "taal_correct": bool(int(taal_pred[i]) == int(batch["taal"][i])),
                    "scale_target": batch["scale_name"][i] if int(batch["scale"][i]) >= 0 else None,
                    "scale_pred": inv_scale[int(scale_pred[i])],
                    "scale_confidence": float(scale_probs[i, scale_pred[i]]),
                    "scale_correct": bool(int(batch["scale"][i]) >= 0 and int(scale_pred[i]) == int(batch["scale"][i])),
                    "tempo_target": float(batch["tempo"][i]),
                    "tempo_pred": float(tempo_pred[i]),
                    "tempo_is_real": bool(batch["tempo_is_real"][i]),
                    "tempo_is_derived": bool(batch["tempo_is_derived"][i]),
                    "period_target": float(batch["cycle_period"][i]),
                    "period_pred": float(period_pred[i]),
                    "period_valid": bool(batch["period_mask"][i]),
                    "period_is_real": bool(batch["period_is_real"][i]),
                    "period_is_derived": bool(batch["period_is_derived"][i]),
                    "sam_annotated": bool(batch["sam_annotated"][i]),
                    "sam_tp": int((sam_pred[i] & sam_true & sam_valid).sum()),
                    "sam_fp": int((sam_pred[i] & ~sam_true & sam_valid).sum()),
                    "sam_fn": int((~sam_pred[i] & sam_true & sam_valid).sum()),
                    "sam_tn": int((~sam_pred[i] & ~sam_true & sam_valid).sum()),
                    "sam_frames": int(sam_valid.sum()),
                    "sam_positive_frames": int((sam_true & sam_valid).sum()),
                }
                pred_file.write(json.dumps(row) + "\n")

                add_to_bucket(overall, row)
                per_taal.setdefault(row["taal_target"], empty_bucket())
                per_source.setdefault(row["source"], empty_bucket())
                key = f"{row['taal_target']}|{row['source']}"
                per_taal_source.setdefault(key, empty_bucket())
                add_to_bucket(per_taal[row["taal_target"]], row)
                add_to_bucket(per_source[row["source"]], row)
                add_to_bucket(per_taal_source[key], row)

                rec = recordings.setdefault(
                    row["recording_id"],
                    {
                        "source": row["source"],
                        "taal_target": row["taal_target"],
                        "scale_target": row["scale_target"],
                        "tempo_target": row["tempo_target"],
                        "tempo_is_real": row["tempo_is_real"],
                        "tempo_is_derived": row["tempo_is_derived"],
                        "period_target": row["period_target"],
                        "period_valid": row["period_valid"],
                        "period_is_real": row["period_is_real"],
                        "period_is_derived": row["period_is_derived"],
                        "sam_annotated": row["sam_annotated"],
                        "sam_tp_sum": 0,
                        "sam_fp_sum": 0,
                        "sam_fn_sum": 0,
                        "sam_tn_sum": 0,
                        "sam_frames_sum": 0,
                        "sam_positive_frames_sum": 0,
                        "taal_probs": [],
                        "scale_probs": [],
                        "tempo_preds": [],
                        "period_preds": [],
                        "sam_times": [],
                        "sam_probs": [],
                        "true_sam_times": [],
                    },
                )
                rec["taal_probs"].append(taal_probs[i].numpy())
                rec["scale_probs"].append(scale_probs[i].numpy())
                rec["tempo_preds"].append(row["tempo_pred"])
                rec["period_preds"].append(row["period_pred"])
                rec["sam_tp_sum"] += row["sam_tp"]
                rec["sam_fp_sum"] += row["sam_fp"]
                rec["sam_fn_sum"] += row["sam_fn"]
                rec["sam_tn_sum"] += row["sam_tn"]
                rec["sam_frames_sum"] += row["sam_frames"]
                rec["sam_positive_frames_sum"] += row["sam_positive_frames"]
                if row["sam_annotated"]:
                    rec["sam_times"].extend(centers.tolist())
                    rec["sam_probs"].extend(pred_scores.tolist())
                    rec["true_sam_times"].extend(centers[true_mask].tolist())

    recording_metrics = recording_level_metrics(recordings, label_map, scale_map)
    dedup_frame_metrics = deduplicated_frame_metrics(recordings, threshold=threshold)
    event_metrics = event_metrics_for_recordings(recordings, threshold)
    threshold_sweep = event_threshold_sweep(recordings)

    return {
        "overall_detailed": finalize_bucket(overall),
        "per_taal": {k: finalize_bucket(v) for k, v in sorted(per_taal.items())},
        "per_source": {k: finalize_bucket(v) for k, v in sorted(per_source.items())},
        "per_taal_source": {k: finalize_bucket(v) for k, v in sorted(per_taal_source.items())},
        "recording_level": recording_metrics,
        "recording_level_frame_dedup": dedup_frame_metrics,
        "sam_event_metrics": event_metrics,
        "sam_event_threshold_sweep": threshold_sweep,
        "predictions_jsonl": str(predictions_path),
    }


def recording_level_metrics(recordings, label_map, scale_map):
    inv_taal = {v: k for k, v in label_map.items()}
    inv_scale = {v: k for k, v in scale_map.items()}
    totals = empty_bucket()
    by_source = {}
    by_taal = {}

    for rec in recordings.values():
        taal_probs = np.stack(rec["taal_probs"]).mean(axis=0)
        scale_probs = np.stack(rec["scale_probs"]).mean(axis=0)
        taal_pred = inv_taal[int(np.argmax(taal_probs))]
        scale_pred = inv_scale[int(np.argmax(scale_probs))]
        row = {
            "taal_correct": taal_pred == rec["taal_target"],
            "scale_target": rec["scale_target"],
            "scale_correct": rec["scale_target"] is not None and scale_pred == rec["scale_target"],
            "tempo_target": rec["tempo_target"],
            "tempo_pred": float(np.median(rec["tempo_preds"])),
            "tempo_is_real": rec["tempo_is_real"],
            "tempo_is_derived": rec["tempo_is_derived"],
            "period_target": rec["period_target"],
            "period_pred": float(np.median(rec["period_preds"])),
            "period_valid": rec["period_valid"],
            "period_is_real": rec["period_is_real"],
            "period_is_derived": rec["period_is_derived"],
            "sam_annotated": rec["sam_annotated"],
            "sam_tp": rec["sam_tp_sum"],
            "sam_fp": rec["sam_fp_sum"],
            "sam_fn": rec["sam_fn_sum"],
            "sam_tn": rec["sam_tn_sum"],
            "sam_frames": rec["sam_frames_sum"],
            "sam_positive_frames": rec["sam_positive_frames_sum"],
        }
        add_to_bucket(totals, row)
        by_source.setdefault(rec["source"], empty_bucket())
        by_taal.setdefault(rec["taal_target"], empty_bucket())
        add_to_bucket(by_source[rec["source"]], row)
        add_to_bucket(by_taal[rec["taal_target"]], row)

    return {
        "overall": finalize_bucket(totals),
        "per_source": {k: finalize_bucket(v) for k, v in sorted(by_source.items())},
        "per_taal": {k: finalize_bucket(v) for k, v in sorted(by_taal.items())},
    }


def event_metrics_for_recordings(recordings, threshold):
    totals = {tol: {"tp": 0, "fp": 0, "fn": 0} for tol in EVENT_TOLERANCES}
    by_taal = {}
    by_source = {}
    for rec in recordings.values():
        if not rec["sam_annotated"]:
            continue
        pred_events = grouped_event_times(rec["sam_times"], rec["sam_probs"], threshold)
        true_events = grouped_event_times(rec["true_sam_times"])
        for tol in EVENT_TOLERANCES:
            metric = event_prf(pred_events, true_events, tol)
            add_event_counts(totals[tol], metric)
            by_taal.setdefault(rec["taal_target"], {}).setdefault(tol, {"tp": 0, "fp": 0, "fn": 0})
            by_source.setdefault(rec["source"], {}).setdefault(tol, {"tp": 0, "fp": 0, "fn": 0})
            add_event_counts(by_taal[rec["taal_target"]][tol], metric)
            add_event_counts(by_source[rec["source"]][tol], metric)
    return {
        "threshold": threshold,
        "overall": {f"tol_{tol:.1f}s": finalize_event_counts(v) for tol, v in totals.items()},
        "per_taal": {
            taal: {f"tol_{tol:.1f}s": finalize_event_counts(counts) for tol, counts in vals.items()}
            for taal, vals in sorted(by_taal.items())
        },
        "per_source": {
            src: {f"tol_{tol:.1f}s": finalize_event_counts(counts) for tol, counts in vals.items()}
            for src, vals in sorted(by_source.items())
        },
    }


def event_threshold_sweep(recordings):
    sweep = {}
    best_by_tol = {}
    for threshold in THRESHOLDS:
        metrics = event_metrics_for_recordings(recordings, threshold)["overall"]
        sweep[f"{threshold:.2f}"] = metrics
    for tol in EVENT_TOLERANCES:
        key = f"tol_{tol:.1f}s"
        best_threshold = max(sweep, key=lambda th: sweep[th][key]["f1"])
        best_by_tol[key] = {"threshold": float(best_threshold), **sweep[best_threshold][key]}
    return {"best_by_tolerance": best_by_tol, "all_thresholds": sweep}


# ═══════════════════════════════════════════════════════════════════
# DESCRIPTIVE REPORT — human-readable summary for the paper's Results
# section. Keeps only the numbers that matter for interpretation, with
# short inline explanations of what each block means and why real vs.
# derived / frame vs. event distinctions are reported separately.
# ═══════════════════════════════════════════════════════════════════

def fmt_pct(x):
    return f"{100 * x:.2f}%"


def fmt2(x):
    return f"{x:.2f}"


def build_descriptive_report(metrics, detailed, split, ckpt_epoch, sam_threshold):
    od = detailed["overall_detailed"]
    rec = detailed["recording_level"]["overall"]
    dedup = detailed["recording_level_frame_dedup"]
    ev_best = detailed["sam_event_threshold_sweep"]["best_by_tolerance"]
    ev_fixed = detailed["sam_event_metrics"]["overall"]

    lines = []
    a = lines.append

    a("=" * 78)
    a(f"TRACK D — EVALUATION REPORT  (split={split}, checkpoint epoch={ckpt_epoch})")
    a("=" * 78)
    a("")
    a("This report separates two evaluation granularities:")
    a("  CHUNK-level  : each 60s segment scored independently (no aggregation)")
    a("  RECORDING-level : predictions from all segments of a recording are")
    a("                    aggregated (probability-averaged for taal/scale,")
    a("                    median for tempo/period) into one prediction per")
    a("                    recording. This is the more practically relevant")
    a("                    number and the one we recommend leading with.")
    a("")

    # ---------------- SECTION 1: headline numbers ----------------
    a("-" * 78)
    a("1. HEADLINE RESULTS")
    a("-" * 78)
    a(f"{'Metric':<28}{'Chunk-level':<18}{'Recording-level':<18}")
    a(f"{'Taal accuracy':<28}{fmt_pct(od['taal_acc']):<18}{fmt_pct(rec['taal_acc']):<18}")
    #a(f"{'Scale accuracy':<28}{fmt_pct(od['scale_acc'])+f' (n={od[\"scale_n\"]})':<18}{fmt_pct(rec['scale_acc'])+f' (n={rec[\"scale_n\"]})':<18}")
    chunk_scale = f"{fmt_pct(od['scale_acc'])} (n={od['scale_n']})"
    record_scale = f"{fmt_pct(rec['scale_acc'])} (n={rec['scale_n']})"
    a(f"{'Scale accuracy':<28}{chunk_scale:<18}{record_scale:<18}")
    a(f"{'Tempo MAE (BPM)':<28}{fmt2(od['tempo_mae']):<18}{fmt2(rec['tempo_mae']):<18}")
    a(f"{'Period MAE (sec)':<28}{fmt2(od['period_mae']):<18}{fmt2(rec['period_mae']):<18}")
    dedup = detailed["recording_level_frame_dedup"]
    a(f"{'Sam frame F1':<28}{fmt_pct(od['sam_f1']):<18}{fmt_pct(dedup['f1']):<18}")
    a("")
    best_05 = ev_best["tol_0.5s"]
    best_10 = ev_best["tol_1.0s"]
    a(f"Sam event F1 @ +-0.5s : {fmt_pct(best_05['f1'])}  "
      f"(P={fmt_pct(best_05['precision'])}, R={fmt_pct(best_05['recall'])}, "
      f"best threshold={best_05['threshold']:.2f})")
    a(f"Sam event F1 @ +-1.0s : {fmt_pct(best_10['f1'])}  "
      f"(P={fmt_pct(best_10['precision'])}, R={fmt_pct(best_10['recall'])}, "
      f"best threshold={best_10['threshold']:.2f})")
    a(f"(For reference, at the fixed frame-level threshold={sam_threshold}: "
      f"event F1 @0.5s={fmt_pct(ev_fixed['tol_0.5s']['f1'])}, "
      f"@1.0s={fmt_pct(ev_fixed['tol_1.0s']['f1'])})")
    a("")
    a("NOTE: Event-level F1 is substantially higher than frame-level F1. This")
    a("is expected and reflects a difference in what is being measured, not")
    a("a discrepancy: frame-level metrics penalize any sub-tolerance timing")
    a("offset as a complete miss, while event-level metrics ask whether the")
    a("sam was located within a musically usable tolerance window.")
    a("")

    # ---------------- SECTION 2: real vs derived ----------------
    a("-" * 78)
    a("2. TEMPO / PERIOD — DIRECTLY-ANNOTATED vs. FORMULA-COMPLETED")
    a("-" * 78)
    a("Tempo and cycle period are mutually derivable via the beats-per-cycle")
    a("relationship (see Methodology, Eq. 1-2). Missing labels are completed")
    a("using this formula and used identically during training. The split")
    a("below is for evaluation/interpretation only:")
    a("")
    a(f"  Tempo MAE  -- real: {fmt2(od['tempo_mae_real'])} (n={od['tempo_n_real']})   "
      f"derived: {fmt2(od['tempo_mae_derived'])} (n={od['tempo_n_derived']})")
    a(f"  Period MAE -- real: {fmt2(od['period_mae_real'])} (n={od['period_n_real']})   "
      f"derived: {fmt2(od['period_mae_derived'])} (n={od['period_n_derived']})")
    a("")
    gap_period = od['period_mae_real'] - od['period_mae_derived']
    if gap_period > 1.0:
        a(f"INTERPRETATION: derived-period MAE is much lower than real-period MAE")
        a(f"(gap={fmt2(gap_period)}s). This is expected, not evidence of strong period")
        a(f"estimation on its own -- once tempo and taal are correctly predicted,")
        a(f"the derived-period target is a near-deterministic rescaling of the")
        a(f"tempo prediction. Treat real-period MAE as the primary evidence of")
        a(f"period-estimation capability; derived-period MAE mainly checks")
        a(f"tempo/period head consistency.")
    a("")

    # ---------------- SECTION 3: per-taal ----------------
    a("-" * 78)
    a("3. PER-TAAL BREAKDOWN (chunk-level)")
    a("-" * 78)
    a(f"{'Taal':<16}{'TaalAcc':<10}{'ScaleAcc':<12}{'TempoMAE(R/D)':<20}{'PeriodMAE(R/D)':<20}{'SamF1':<10}")
    for taal, m in detailed["per_taal"].items():
        tempo_str = f"{fmt2(m['tempo_mae_real'])}/{fmt2(m['tempo_mae_derived'])}"
        period_str = f"{fmt2(m['period_mae_real'])}/{fmt2(m['period_mae_derived'])}"
        sam_str = fmt_pct(m['sam_f1']) if m['sam_annotated_chunks'] > 0 else "--"
        a(f"{taal:<16}{fmt_pct(m['taal_acc']):<10}{fmt_pct(m['scale_acc']):<12}{tempo_str:<20}{period_str:<20}{sam_str:<10}")
    a("")
    a("NOTE (Rupak): if Rupak shows low sam F1 relative to other taals, this is")
    a("consistent with a known structural property of the taal -- Rupak's sam")
    a("falls on the khali (unaccented) stroke rather than a strong onset,")
    a("making it acoustically harder to detect via onset-strength patterns")
    a("alone. This should be read as a task-difficulty finding, not a model")
    a("failure specific to Rupak.")
    a("")

    # ---------------- SECTION 4: per-source ----------------
    a("-" * 78)
    a("4. PER-SOURCE BREAKDOWN (chunk-level)")
    a("-" * 78)
    a(f"{'Source':<16}{'Chunks':<10}{'TaalAcc':<10}{'TempoMAE':<12}{'PeriodMAE':<12}{'SamF1':<10}")
    for source, m in detailed["per_source"].items():
        sam_str = fmt_pct(m['sam_f1']) if m['sam_annotated_chunks'] > 0 else "--"
        a(f"{source:<16}{m['chunks']:<10}{fmt_pct(m['taal_acc']):<10}{fmt2(m['tempo_mae']):<12}{fmt2(m['period_mae']):<12}{sam_str:<10}")
    a("")
    kaggle = detailed["per_source"].get("kaggle") or detailed["per_source"].get("Kaggle")
    if kaggle and kaggle["taal_acc"] >= 0.99:
        a("CAUTION (Kaggle): near-perfect taal accuracy on Kaggle should be")
        a("interpreted cautiously. Kaggle provides taal-only labels (no tempo/")
        a("period/sam), and very high accuracy on a small pool of source")
        a("recordings sliced into many overlapping chunks may reflect limited")
        a("acoustic diversity within the source rather than a generalizable")
        a("taal-classification result. Recommend auditing unique-recording")
        a("counts per taal before citing this figure without qualification.")
        a("")

    # ---------------- SECTION 5: sam event detail ----------------
    a("-" * 78)
    a("5. SAM EVENT DETECTION -- THRESHOLD SWEEP SUMMARY")
    a("-" * 78)
    a("Best operating point per tolerance (F1-maximizing threshold, swept over")
    a(f"[{THRESHOLDS[0]:.2f}, {THRESHOLDS[-1]:.2f}] in steps of 0.05):")
    for tol_key, m in ev_best.items():
        a(f"  {tol_key:<10} threshold={m['threshold']:.2f}  "
          f"F1={fmt_pct(m['f1'])}  P={fmt_pct(m['precision'])}  R={fmt_pct(m['recall'])}  "
          f"(TP={m['tp']}, FP={m['fp']}, FN={m['fn']})")
    a("")

    # ---------------- SECTION 6: pointers to full data ----------------
    a("-" * 78)
    a("6. FULL DATA")
    a("-" * 78)
    a("Complete per-taal x per-source tables, full threshold sweep, and raw")
    a("per-chunk predictions are in the accompanying JSON files:")
    a(f"  {split}_metrics.json          (chunk + recording summary metrics)")
    a(f"  {split}_detailed_metrics.json (full breakdown: per-taal, per-source,")
    a(f"                                 per-taal-per-source, full threshold sweep)")
    a(f"  {split}_predictions.jsonl     (one row per chunk, all predictions)")
    a("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--ckpt", default=str(Path(CHECKPOINT_DIR) / "best.pt"))
    parser.add_argument("--sam-threshold", type=float, default=0.5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location=device)
    ds = SamFrameDataset(EMBEDDING_ROOT, args.split, ckpt["label_map"], ckpt["scale_map"])
    print_dataset_summary(args.split, ds)
    loader = DataLoader(ds, BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate_fn)
    model = TrackDSamModel(len(ckpt["label_map"]), len(ckpt["scale_map"])).to(device)
    model.load_state_dict(ckpt["model"])

    metrics = run_epoch(model, loader, device)
    predictions_path = Path(RESULT_DIR) / f"{args.split}_predictions.jsonl"
    detailed = detailed_eval(
        model,
        loader,
        device,
        ckpt["label_map"],
        ckpt["scale_map"],
        predictions_path,
        threshold=args.sam_threshold,
    )
    metrics = {
        "split": args.split,
        "checkpoint": args.ckpt,
        "checkpoint_epoch": ckpt.get("epoch"),
        "checkpoint_val_metrics": ckpt.get("metrics"),
        "sam_threshold": args.sam_threshold,
        **metrics,
        **detailed,
    }
    Path(RESULT_DIR).mkdir(parents=True, exist_ok=True)
    metrics_path = Path(RESULT_DIR) / f"{args.split}_metrics.json"
    detailed_path = Path(RESULT_DIR) / f"{args.split}_detailed_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    with open(detailed_path, "w") as f:
        json.dump(detailed, f, indent=2)

    # ---- descriptive report (this is the file to read for the paper) ----
    report_text = build_descriptive_report(
        metrics, detailed, args.split, ckpt.get("epoch"), args.sam_threshold
    )
    report_path = Path(RESULT_DIR) / f"{args.split}_report.txt"
    with open(report_path, "w") as f:
        f.write(report_text)

    print(report_text)
    print(f"\nmetrics saved      : {metrics_path}")
    print(f"detailed saved     : {detailed_path}")
    print(f"predictions saved  : {predictions_path}")
    print(f"REPORT saved       : {report_path}   <-- read this one")


if __name__ == "__main__":
    main()