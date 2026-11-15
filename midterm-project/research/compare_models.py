"""Train the best version of every model from the notebook and write research/model_comparison.xlsx.

The notebook (research/notebook.ipynb) tunes each model family. This script takes the best settings
of each family, trains them the same way, and scores them with the metrics from the course:

  * accuracy (lesson 4.1) and the dummy model (4.2)
  * confusion table (4.3), precision / recall (4.4), F1
  * ROC AUC (4.5-4.6)
  * K-fold cross-validation (4.7), here with guitarists as folds

plus the project's own metrics (tab / pitch F1, string choice, note F1). Each model is trained on
100,000 frames of 4 guitarists and scored on all frames of the 5th, once for each of the 5 non-test
guitarists. The fold of the validation guitarist is the notebook's validation result.
The last sheet scores the production model (production/model.bin) on the test guitarist.

Usage (from the project folder, after `uv run python data/get_data.py`):
    uv run python research/compare_models.py            # ~45 minutes on a 12-core CPU
    N_TRAIN=3000 uv run python research/compare_models.py   # quick check (~2 minutes)
"""

import os
import pickle
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "production"))
import guitar as G  # noqa: E402
from train import load_data, note_scores, split_players, true_notes_by_track  # noqa: E402

N_TRAIN = int(os.environ.get("N_TRAIN", 100_000))
OUTPUT = Path(__file__).resolve().parent / "model_comparison.xlsx"
NOTE_MIN_FRAMES = 4  # the note clean-up of the production model, applied to every model for note F1

# Best settings of each family and its decision threshold, both tuned in the notebook (sections 5-6).
MODELS = {
    "Dummy (always silent)": (None, None, "predicts 'not playing' for every string"),
    "Logistic regression": (
        lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000)),
        0.25, "C=1.0, standardised features"),
    "Decision tree": (
        lambda: DecisionTreeClassifier(max_depth=15, min_samples_leaf=50, random_state=1),
        0.25, "max_depth=15, min_samples_leaf=50"),
    "Random forest": (
        lambda: RandomForestClassifier(n_estimators=50, max_depth=None, min_samples_leaf=5, n_jobs=2,
                                       random_state=1),
        0.30, "n_estimators=50, max_depth=None, min_samples_leaf=5"),
    "XGBoost": (
        lambda: XGBClassifier(n_estimators=200, learning_rate=0.3, max_depth=6, min_child_weight=30,
                              tree_method="hist", max_bin=64, n_jobs=2, random_state=1),
        0.15, "n_estimators=200, eta=0.3, max_depth=6, min_child_weight=30"),
}
SELECTED = "XGBoost"


def scores(y_true, proba, threshold, track, true_notes) -> dict:
    """All metrics for one model on one set of frames."""
    pred = G.decide(proba, threshold)
    tab = G.tab_metrics(y_true, pred)
    playing_true, playing_pred = y_true > 0, pred > 0
    tp = int((playing_true & playing_pred).sum())
    fp = int((~playing_true & playing_pred).sum())
    fn = int((playing_true & ~playing_pred).sum())
    tn = int((~playing_true & ~playing_pred).sum())
    det_p = tp / (tp + fp) if tp + fp else 0.0
    det_r = tp / (tp + fn) if tp + fn else 0.0
    p_playing = 1.0 - proba[:, :, 0]
    auc = np.mean([roc_auc_score(playing_true[:, s], p_playing[:, s]) for s in range(G.N_STRINGS)])
    notes = note_scores(pred, track, true_notes, NOTE_MIN_FRAMES, 0)
    return {
        "Accuracy": tab["accuracy"],
        "ROC AUC": auc,
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "Detection precision": det_p,
        "Detection recall": det_r,
        "Detection F1": 2 * det_p * det_r / (det_p + det_r) if det_p + det_r else 0.0,
        "Tab precision": tab["tab_precision"],
        "Tab recall": tab["tab_recall"],
        "Tab F1": tab["tab_f1"],
        "Pitch F1": tab["pitch_f1"],
        "String choice": tab["string_choice"],
        "Note F1": notes["note_f1"],
        **{f"Tab F1, {name} string": G.tab_metrics(y_true[:, [s]], pred[:, [s]])["tab_f1"]
           for s, name in enumerate(G.STRING_NAMES)},
    }


def main():
    X, y, track, tracks, notes = load_data()
    test_player, val_player = split_players()
    players = tracks["player"].to_numpy()[track]
    folds = sorted(set(range(6)) - {test_player})
    print(f"training frames per model: {N_TRAIN:,}; folds (held-out guitarists): {folds}; "
          f"validation guitarist: {val_player}; test guitarist: {test_player}")

    # Results are saved after every model, so an interrupted run continues where it stopped.
    cache = ROOT / "data" / "processed" / f"compare_models_{N_TRAIN}.csv"
    done = pd.read_csv(cache).to_dict("records") if cache.exists() else []
    rows = list(done)
    for fold in folds:
        fit_pool = np.flatnonzero((players != test_player) & (players != fold))
        fit_idx = np.sort(np.random.default_rng(1).choice(fit_pool, size=N_TRAIN, replace=False))
        eval_mask = players == fold
        true_notes = true_notes_by_track(notes, np.unique(track[eval_mask]))
        for name, (make, threshold, _) in MODELS.items():
            if any(r["Model"] == name and r["Held-out guitarist"] == fold for r in done):
                continue
            start = time.time()
            if make is None:
                proba = np.zeros((int(eval_mask.sum()), G.N_STRINGS, G.N_CLASSES), dtype=np.float32)
                proba[:, :, 0] = 1.0
                threshold, fit_seconds, size_mb = 0.5, 0, 0.0
            else:
                model = G.StringwiseClassifier(make()).fit(X[fit_idx], y[fit_idx])
                fit_seconds = time.time() - start
                size_mb = len(pickle.dumps(model)) / 1e6
                proba = model.predict_proba(X[eval_mask])
                del model
            row = {"Model": name, "Held-out guitarist": fold,
                   **scores(y[eval_mask], proba, threshold, track[eval_mask], true_notes),
                   "Fit seconds": round(fit_seconds), "Model size (MB)": round(size_mb, 1)}
            rows.append(row)
            pd.DataFrame(rows).to_csv(cache, index=False)
            print(f"  guitarist {fold}  {name:22s} tab F1 {row['Tab F1']:.3f}  AUC {row['ROC AUC']:.3f}  "
                  f"({time.time() - start:.0f} s)", flush=True)
    cv = pd.DataFrame(rows)

    # --- Summary: validation guitarist + cross-validation mean/std -----------------------------
    val = cv[cv["Held-out guitarist"] == val_player].set_index("Model")
    summary_cols = ["Accuracy", "ROC AUC", "Detection precision", "Detection recall", "Detection F1",
                    "Tab precision", "Tab recall", "Tab F1", "Pitch F1", "String choice", "Note F1"]
    summary = val[summary_cols].copy()
    grouped = cv.groupby("Model", sort=False)
    summary.insert(0, "Settings", [MODELS[m][2] for m in summary.index])
    summary.insert(1, "Threshold", [MODELS[m][1] for m in summary.index])
    summary["CV tab F1, mean"] = grouped["Tab F1"].mean()
    summary["CV tab F1, std"] = grouped["Tab F1"].std()
    summary["CV ROC AUC, mean"] = grouped["ROC AUC"].mean()
    summary["CV note F1, mean"] = grouped["Note F1"].mean()
    summary["Fit seconds"] = val["Fit seconds"]
    summary["Model size (MB)"] = val["Model size (MB)"]
    summary["Selected"] = ["yes" if m == SELECTED else "" for m in summary.index]
    summary = summary.reset_index()

    confusion = val[["TP", "FN", "FP", "TN"]].copy()
    confusion["True positive rate (recall)"] = confusion["TP"] / (confusion["TP"] + confusion["FN"])
    confusion["False positive rate"] = confusion["FP"] / (confusion["FP"] + confusion["TN"])
    confusion = confusion.reset_index()

    per_string = val[[f"Tab F1, {n} string" for n in G.STRING_NAMES]].reset_index()
    cv_table = cv[["Model", "Held-out guitarist", "Accuracy", "ROC AUC", "Tab precision", "Tab recall",
                   "Tab F1", "Pitch F1", "String choice", "Note F1", "Fit seconds"]]

    # --- Production model on the test guitarist ------------------------------------------------
    bundle = joblib.load(ROOT / "production" / "model.bin")
    test_mask = players == test_player
    test_proba = bundle["model"].predict_proba(X[test_mask])
    test_true_notes = true_notes_by_track(notes, np.unique(track[test_mask]))
    test = scores(y[test_mask], test_proba, bundle["threshold"], track[test_mask], test_true_notes)
    final = pd.DataFrame([
        ("Model", f"XGBoost, {MODELS['XGBoost'][2]}"),
        ("Training frames", f"{bundle['n_train_frames']:,} (guitarists {', '.join(map(str, folds))})"),
        ("Test guitarist", test_player),
        ("Threshold (tuned on validation by note F1)", bundle["threshold"]),
        ("Note clean-up: shortest note (frames)", bundle["min_note_frames"]),
        ("Accuracy", test["Accuracy"]), ("ROC AUC", test["ROC AUC"]),
        ("Detection precision", test["Detection precision"]), ("Detection recall", test["Detection recall"]),
        ("Tab precision (frames, before clean-up)", test["Tab precision"]),
        ("Tab recall (frames, before clean-up)", test["Tab recall"]),
        ("Tab F1 (frames, before clean-up)", test["Tab F1"]),
        ("Tab F1 (frames, after clean-up)", bundle["test_scores"]["frame"]["tab_f1"]),
        ("Pitch F1 (frames, after clean-up)", bundle["test_scores"]["frame"]["pitch_f1"]),
        ("String choice (frames, after clean-up)", bundle["test_scores"]["frame"]["string_choice"]),
        ("Note precision", bundle["test_scores"]["note"]["note_precision"]),
        ("Note recall", bundle["test_scores"]["note"]["note_recall"]),
        ("Note F1", bundle["test_scores"]["note"]["note_f1"]),
    ], columns=["Metric", "Value"])

    definitions = pd.DataFrame([
        ("Frame", "One ~23 ms time step. Every metric except the note metrics is computed over all "
                  "frame x string cells of the held-out guitarist."),
        ("Accuracy", "Share of frame x string cells where the predicted class (silent, open, fret 1..19) is "
                     "exactly right. Dominated by silent cells: the dummy model already scores high."),
        ("ROC AUC", "How well P(string playing) separates playing from silent cells, averaged over the 6 strings."),
        ("TP / FP / FN / TN", "Confusion table of 'string playing' vs 'silent', summed over strings."),
        ("Detection precision / recall / F1", "Same as above: is the string playing at all, ignoring the fret."),
        ("Tab precision / recall / F1", "A predicted note counts only if the string AND fret are right. "
                                        "Main metric for model selection (Tab F1)."),
        ("Pitch F1", "A predicted note counts if its pitch is sounding, whatever string it was put on."),
        ("String choice", "Of the correctly detected pitches, the share also on the correct string."),
        ("Note F1", f"After joining frames into notes (notes shorter than {NOTE_MIN_FRAMES} frames dropped): a note "
                    "is correct if string and fret are right and it starts within 46 ms of a real note."),
        ("Threshold", "A string is predicted as playing when P(playing) >= threshold; tuned per model on the "
                      "validation guitarist in the notebook."),
        ("CV", "Leave-one-guitarist-out cross-validation over the 5 non-test guitarists; mean and standard "
               "deviation over the 5 folds."),
        ("Validation / test guitarist", f"Validation = guitarist {val_player} (used for tuning), test = "
                                        f"guitarist {test_player} (used once, for the final model)."),
    ], columns=["Term", "Meaning"])

    sheets = {
        "Summary": summary,
        "Confusion tables": confusion,
        "Per string": per_string,
        "Cross-validation": cv_table,
        "Final model (test)": final,
        "Metric definitions": definitions,
    }
    write_excel(sheets, OUTPUT)
    print(f"\nwrote {OUTPUT.relative_to(ROOT)}")
    print(summary[["Model", "Accuracy", "ROC AUC", "Tab F1", "CV tab F1, mean", "Note F1"]].round(3).to_string(index=False))


def write_excel(sheets: dict, path: Path) -> None:
    from openpyxl.styles import Alignment, Font, PatternFill

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name, index=False)
            ws = writer.sheets[name]
            ws.freeze_panes = "B2"
            for cell in ws[1]:
                cell.font = Font(bold=True)
                cell.fill = PatternFill("solid", fgColor="E4E3DF")
                cell.alignment = Alignment(wrap_text=True, vertical="top")
            for col in ws.columns:
                header = str(col[0].value)
                width = max(len(header) if len(header) < 18 else 18,
                            *(len(str(c.value)) for c in col[1:] if c.value is not None and not isinstance(c.value, float)),
                            8)
                ws.column_dimensions[col[0].column_letter].width = min(width + 2, 70)
                for c in col[1:]:
                    if isinstance(c.value, float):
                        c.number_format = "0.000"
                    if name == "Metric definitions":
                        c.alignment = Alignment(wrap_text=True, vertical="top")
            if name == "Summary":
                for row in ws.iter_rows(min_row=2):
                    if row[-1].value == "yes":
                        for c in row:
                            c.font = Font(bold=True)


if __name__ == "__main__":
    main()
