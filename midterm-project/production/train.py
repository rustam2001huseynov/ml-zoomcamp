"""Train the final guitar-tab model and save it to model.bin.

Reproduces the model selected in research/notebook.ipynb (sections 6-7) and adds the note clean-up:

1. Validation stage: train XGBoost on 100,000 frames of the training guitarists (exactly the tuning
   model of the notebook) and predict the validation guitarist. On these predictions, tune the
   decision threshold together with the note clean-up (shortest note to keep, longest silent gap to
   fill), maximising *note-level* F1: a tab shows notes, so that is what the clean-up should optimise.
2. Final stage: retrain with the same settings on 200,000 frames of the training + validation
   guitarists, report frame- and note-level scores on the test guitarist, and save everything the
   web service needs (model + threshold + clean-up settings) with joblib.

Run `uv run python data/get_data.py` first. Usage (from the project folder):
    uv run python production/train.py                  # full run, ~15 minutes on a 12-core CPU
    uv run python production/train.py --n-train 5000 --output model_test.bin   # quick check
"""

import argparse
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

import guitar as F
import guitar as tab
from guitar import StringwiseClassifier, decide, tab_metrics

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"

# Selected in research/notebook.ipynb (sections 6.3 and 7)
XGB_PARAMS = dict(n_estimators=200, learning_rate=0.3, max_depth=6, min_child_weight=30,
                  tree_method="hist", max_bin=64, n_jobs=2, random_state=1)

# Clean-up search space (frames are ~23 ms)
THRESHOLDS = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]
MIN_NOTE_FRAMES = [1, 2, 3, 4, 5, 6, 8]
MAX_GAP_FRAMES = [0, 1, 2, 3]


def load_data():
    data = np.load(DATA_DIR / "frames.npz")
    tracks = pd.read_csv(DATA_DIR / "tracks.csv", index_col="track_id")
    notes = pd.read_csv(DATA_DIR / "notes.csv")
    X = F.add_deltas(data["X"].astype(np.float32), data["track"])
    return X, data["y"], data["track"], tracks, notes


def split_players():
    """Same guitarist split as the notebook: one test and one validation guitarist."""
    rng = np.random.default_rng(1)
    test_player, val_player = rng.permutation(6)[:2]
    return int(test_player), int(val_player)


def true_notes_by_track(notes: pd.DataFrame, track_ids) -> dict:
    """Annotated notes in frames (same rounding as the frame labels in data/get_data.py)."""
    fs = F.frame_seconds()
    result = {}
    for t in track_ids:
        result[t] = [
            {"string": int(n.string), "fret": int(n.fret), "start": int(np.ceil(n.start / fs)),
             "end": int(np.ceil((n.start + n.duration) / fs))}
            for n in notes[notes["track_id"] == t].itertuples()
        ]
    return result


def note_scores(pred_classes, track, true_notes, min_frames, max_gap) -> dict:
    """Note-level P/R/F1 summed over recordings (notes never cross recording boundaries)."""
    total = {"tp": 0, "n_pred": 0, "n_true": 0}
    for t, true in true_notes.items():
        pred = tab.frames_to_notes(pred_classes[track == t], min_frames, max_gap)
        for k, v in tab.note_metrics(true, pred).items():
            total[k] += v
    return tab.prf(**total)


def clean_by_track(pred_classes, track, min_frames, max_gap):
    cleaned = np.zeros_like(pred_classes)
    for t in np.unique(track):
        mask = track == t
        cleaned[mask] = tab.clean_frames(pred_classes[mask], min_frames, max_gap)
    return cleaned


def fit(X, y, idx):
    model = StringwiseClassifier(XGBClassifier(**XGB_PARAMS))
    start = time.time()
    model.fit(X[idx], y[idx])
    print(f"  trained on {len(idx):,} frames in {time.time() - start:.0f} s")
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-train", type=int, default=100_000, help="frames for the validation stage")
    parser.add_argument("--output", default=str(Path(__file__).resolve().parent / "model.bin"))
    args = parser.parse_args()
    n_final = 2 * args.n_train

    X, y, track, tracks, notes = load_data()
    test_player, val_player = split_players()
    players = tracks["player"].to_numpy()[track]
    train_mask = (players != test_player) & (players != val_player)
    val_mask, test_mask = players == val_player, players == test_player
    print(f"frames: {len(X):,}  test guitarist: {test_player}  validation guitarist: {val_player}")

    # --- 1. Validation stage --------------------------------------------------------------------
    print("\n1. Tuning threshold and note clean-up on the validation guitarist")
    train_idx = np.sort(np.random.default_rng(1).choice(np.flatnonzero(train_mask), size=args.n_train,
                                                         replace=False))
    model = fit(X, y, train_idx)
    val_proba = model.predict_proba(X[val_mask])
    val_track = track[val_mask]
    val_true_notes = true_notes_by_track(notes, np.unique(val_track))

    rows = []
    for t in THRESHOLDS:
        pred = decide(val_proba, t)
        for min_frames in MIN_NOTE_FRAMES:
            for max_gap in MAX_GAP_FRAMES:
                scores = note_scores(pred, val_track, val_true_notes, min_frames, max_gap)
                rows.append({"threshold": t, "min_note_frames": min_frames, "max_gap_frames": max_gap,
                             **scores})
    grid = pd.DataFrame(rows).sort_values("note_f1", ascending=False)
    best = grid.iloc[0]
    threshold, min_frames, max_gap = float(best["threshold"]), int(best["min_note_frames"]), int(best["max_gap_frames"])

    # Compare with no clean-up at the frame-level threshold from the notebook.
    no_cleanup = grid[(grid["threshold"] == 0.15) & (grid["min_note_frames"] == 1) & (grid["max_gap_frames"] == 0)]
    print("  top settings by note F1:")
    print(grid.head(5).round(3).to_string(index=False))
    print("  without clean-up (threshold 0.15):")
    print(no_cleanup.round(3).to_string(index=False))

    # --- 2. Final model -------------------------------------------------------------------------
    print(f"\n2. Final model on training + validation guitarists "
          f"(threshold {threshold}, min note {min_frames} frames, max gap {max_gap} frames)")
    final_idx = np.sort(np.random.default_rng(1).choice(np.flatnonzero(~test_mask), size=n_final, replace=False))
    final_model = fit(X, y, final_idx)

    test_pred = decide(final_model.predict_proba(X[test_mask]), threshold)
    test_track = track[test_mask]
    test_true_notes = true_notes_by_track(notes, np.unique(test_track))
    test_cleaned = clean_by_track(test_pred, test_track, min_frames, max_gap)
    report = pd.DataFrame({
        "frames, raw": tab_metrics(y[test_mask], test_pred),
        "frames, cleaned": tab_metrics(y[test_mask], test_cleaned),
    }).loc[["tab_precision", "tab_recall", "tab_f1", "pitch_f1", "string_choice"]]
    notes_raw = note_scores(test_pred, test_track, test_true_notes, 1, 0)
    notes_clean = note_scores(test_pred, test_track, test_true_notes, min_frames, max_gap)
    print("  test guitarist, frame level:")
    print(report.round(3).to_string())
    print("  test guitarist, note level (string + fret + onset within 46 ms):")
    print(pd.DataFrame({"raw": notes_raw, "cleaned": notes_clean}).round(3).to_string())

    bundle = {
        "model": final_model,
        "threshold": threshold,
        "min_note_frames": min_frames,
        "max_gap_frames": max_gap,
        "xgb_params": XGB_PARAMS,
        "n_train_frames": n_final,
        "test_scores": {"frame": report["frames, cleaned"].to_dict(), "note": notes_clean},
    }
    joblib.dump(bundle, args.output, compress=3)
    print(f"\nsaved {args.output} ({Path(args.output).stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
