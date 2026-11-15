"""Everything the project shares between data preparation, research, training and the web service.

Keeping it in one module guarantees that the service computes features, decisions and notes exactly
the way they were computed for training and evaluation.

Sections:
  1. Audio features: CQT spectrogram (+ deltas), guitar constants, labels
  2. Metrics: tab / pitch precision, recall, F1, string choice, and the decision threshold
  3. StringwiseClassifier: one classifier per string, behaving like a single model
  4. Frames -> notes -> ASCII tab (+ note-level metrics, optional duplicate-pitch clean-up)

Labels and predictions are (n_frames, 6) arrays of classes per string, low E first:
0 = not playing, 1 = open string, 2..20 = frets 1..19.
"""

from pathlib import Path

import librosa
import numpy as np
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, clone
from sklearn.pipeline import Pipeline


# ==================================================================================================
# 1. Audio features
# ==================================================================================================

# --- Audio / spectrogram settings -------------------------------------------
SAMPLE_RATE = 22050          # resample everything to this rate
HOP_LENGTH = 512             # 512 / 22050 = ~23.2 ms per frame (~43 frames/sec)
FMIN = librosa.note_to_hz("E2")   # lowest note of a guitar in standard tuning
BINS_PER_OCTAVE = 24         # 2 bins per semitone
N_BINS = 24 * 7              # 7 octaves: E2 (82 Hz) .. ~E9 (10.5 kHz), under Nyquist
TOP_DB = 80.0                # dB range kept after log scaling

# --- Guitar settings ---------------------------------------------------------
# Standard tuning, low to high: E2 A2 D3 G3 B3 E4 (MIDI note numbers).
# Index 0 is the low E string, matching GuitarSet's string numbering.
OPEN_STRING_MIDI = np.array([40, 45, 50, 55, 59, 64])
STRING_NAMES = ["E", "A", "D", "G", "B", "e"]
N_STRINGS = 6
MAX_FRET = 19
# Classes per string: 0 = not playing, 1 = open string, 2..20 = frets 1..19
N_CLASSES = MAX_FRET + 2


def frame_seconds() -> float:
    return HOP_LENGTH / SAMPLE_RATE


def load_audio(path: str | Path) -> np.ndarray:
    """Load any audio file as mono float32 at SAMPLE_RATE."""
    y, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    return y.astype(np.float32)


def compute_cqt(y: np.ndarray) -> np.ndarray:
    """Log-magnitude Constant-Q transform.

    Returns an array of shape (n_frames, N_BINS) with values in [-TOP_DB, 0] dB,
    relative to the loudest bin of the clip (so the input gain doesn't matter).
    """
    C = librosa.cqt(
        y,
        sr=SAMPLE_RATE,
        hop_length=HOP_LENGTH,
        fmin=FMIN,
        n_bins=N_BINS,
        bins_per_octave=BINS_PER_OCTAVE,
    )
    C_db = librosa.amplitude_to_db(np.abs(C), ref=np.max, top_db=TOP_DB)
    return C_db.T.astype(np.float32)


def add_deltas(X: np.ndarray, track: np.ndarray | None = None) -> np.ndarray:
    """Append the change since the previous frame to every frame: [X_t, X_t - X_(t-1)].

    The deltas are large when a note starts (onset) and small while it rings. `track` gives the
    recording each frame belongs to, so differences are never taken across two recordings; the
    first frame of each recording gets a delta of 0. Use track=None for a single recording.
    """
    delta = np.zeros_like(X)
    delta[1:] = X[1:] - X[:-1]
    if track is not None:
        delta[1:][track[1:] != track[:-1]] = 0.0
    else:
        delta[0] = 0.0
    return np.hstack([X, delta])


def cqt_bin_frequencies() -> np.ndarray:
    """Centre frequency (Hz) of every CQT bin, for plots and feature names."""
    return librosa.cqt_frequencies(N_BINS, fmin=FMIN, bins_per_octave=BINS_PER_OCTAVE)


def fret_to_class(fret: int) -> int:
    return fret + 1


def class_to_fret(cls: int) -> int | None:
    """None means the string is not playing."""
    return None if cls == 0 else cls - 1


# ==================================================================================================
# 2. Metrics (frame level)
# ==================================================================================================

# * tab precision / recall / F1: a predicted note counts only if the string AND fret are right
# * pitch precision / recall / F1: a predicted note counts if its pitch is sounding in that frame,
#   whatever string it was assigned to (what a pitch detector would be scored on)
# * string choice: of the notes whose pitch was detected correctly, the share also on the right string

def _prf(tp: int, n_pred: int, n_true: int) -> tuple[float, float, float]:
    precision = tp / n_pred if n_pred else 0.0
    recall = tp / n_true if n_true else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def classes_to_pitch(y: np.ndarray) -> np.ndarray:
    """(n, 6) classes -> (n, 6) MIDI pitch, -1 where the string is not playing."""
    return np.where(y > 0, OPEN_STRING_MIDI[None, :] + y - 1, -1)


def pitch_hits(y_true: np.ndarray, y_pred: np.ndarray) -> int:
    """Number of predicted pitches that are sounding in the same frame (multiset matching)."""
    p_true, p_pred = classes_to_pitch(y_true), classes_to_pitch(y_pred)
    # Pitch range on a guitar is 40..83, so a per-frame histogram over 128 MIDI notes is enough.
    rows = np.arange(len(y_true))[:, None]
    h_true = np.zeros((len(y_true), 128), dtype=np.int16)
    h_pred = np.zeros((len(y_true), 128), dtype=np.int16)
    np.add.at(h_true, (np.broadcast_to(rows, p_true.shape)[p_true >= 0], p_true[p_true >= 0]), 1)
    np.add.at(h_pred, (np.broadcast_to(rows, p_pred.shape)[p_pred >= 0], p_pred[p_pred >= 0]), 1)
    return int(np.minimum(h_true, h_pred).sum())


def tab_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    active_true = y_true > 0
    active_pred = y_pred > 0
    n_true, n_pred = int(active_true.sum()), int(active_pred.sum())

    tab_tp = int((active_pred & (y_pred == y_true)).sum())
    tab_p, tab_r, tab_f1 = _prf(tab_tp, n_pred, n_true)

    pitch_tp = pitch_hits(y_true, y_pred)
    pitch_p, pitch_r, pitch_f1 = _prf(pitch_tp, n_pred, n_true)

    return {
        "tab_precision": tab_p,
        "tab_recall": tab_r,
        "tab_f1": tab_f1,
        "pitch_precision": pitch_p,
        "pitch_recall": pitch_r,
        "pitch_f1": pitch_f1,
        "string_choice": tab_tp / pitch_tp if pitch_tp else 0.0,
        "accuracy": float((y_pred == y_true).mean()),
    }


def decide(proba: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Turn (n, 6, 21) probabilities into (n, 6) classes.

    A string is predicted as playing when P(playing) = 1 - P(silent) >= threshold; its fret is then
    the most likely of the 20 playing classes. threshold=0.5 is close to a plain argmax.
    """
    playing = 1.0 - proba[:, :, 0] >= threshold
    fret_class = proba[:, :, 1:].argmax(axis=2) + 1
    return np.where(playing, fret_class, 0).astype(np.int8)


# ==================================================================================================
# 3. One classifier per string
# ==================================================================================================

def _fit_one(estimator, X, y_string, sample_weight=None):
    classes = np.unique(y_string)
    # Re-number the classes present to 0..k-1 (XGBoost requires contiguous labels).
    y_encoded = np.searchsorted(classes, y_string)
    kwargs = {}
    if sample_weight is not None:
        # A Pipeline only accepts weights addressed to one of its steps: give them to the model.
        if isinstance(estimator, Pipeline):
            kwargs = {f"{estimator.steps[-1][0]}__sample_weight": sample_weight}
        else:
            kwargs = {"sample_weight": sample_weight}
    estimator.fit(X, y_encoded, **kwargs)
    return estimator, classes


class StringwiseClassifier(BaseEstimator):
    def __init__(self, estimator, class_weight=None, n_jobs=6):
        """
        estimator:    classifier to clone for each string
        class_weight: None or "balanced" (applied through sample weights, so it works the same
                      for every model type)
        n_jobs:       how many strings to train in parallel
        """
        self.estimator = estimator
        self.class_weight = class_weight
        self.n_jobs = n_jobs

    def _sample_weight(self, y_string):
        if self.class_weight is None:
            return None
        if self.class_weight != "balanced":
            raise ValueError("class_weight must be None or 'balanced'")
        classes, counts = np.unique(y_string, return_counts=True)
        weights = len(y_string) / (len(classes) * counts)
        return weights[np.searchsorted(classes, y_string)]

    def fit(self, X, y):
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_one)(clone(self.estimator), X, y[:, s], self._sample_weight(y[:, s]))
            for s in range(N_STRINGS)
        )
        self.estimators_ = [est for est, _ in results]
        self.classes_ = [cls for _, cls in results]
        return self

    def predict_proba(self, X):
        proba = np.zeros((len(X), N_STRINGS, N_CLASSES), dtype=np.float32)
        for s, (est, classes) in enumerate(zip(self.estimators_, self.classes_)):
            proba[:, s, classes] = est.predict_proba(X)
        return proba

    def predict(self, X):
        proba = self.predict_proba(X)
        return proba.argmax(axis=2).astype(np.int8)


# ==================================================================================================
# 4. Frames -> notes -> tab
# ==================================================================================================

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def _runs(column: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Start, end (exclusive) and value of every run of equal values in a 1-D array."""
    if len(column) == 0:
        return np.array([], int), np.array([], int), np.array([], column.dtype)
    change = np.flatnonzero(np.diff(column)) + 1
    starts = np.r_[0, change]
    ends = np.r_[change, len(column)]
    return starts, ends, column[starts]


def _fill_gaps(column: np.ndarray, max_gap: int) -> np.ndarray:
    """Fill silent gaps of at most max_gap frames between two runs of the same fret."""
    column = column.copy()
    if max_gap <= 0:
        return column
    starts, ends, values = _runs(column)
    for i in range(1, len(values) - 1):
        if values[i] == 0 and ends[i] - starts[i] <= max_gap and values[i - 1] == values[i + 1] != 0:
            column[starts[i]:ends[i]] = values[i - 1]
    return column


def frames_to_notes(classes: np.ndarray, min_frames: int = 1, max_gap: int = 0) -> list[dict]:
    """(n_frames, 6) classes of one recording -> notes sorted by start.

    Each note is {"string": 0..5 (0 = low E), "fret": 0..19, "start": frame, "end": frame (exclusive)}.
    """
    notes = []
    for s in range(N_STRINGS):
        column = _fill_gaps(classes[:, s], max_gap)
        starts, ends, values = _runs(column)
        for start, end, value in zip(starts, ends, values):
            if value > 0 and end - start >= min_frames:
                notes.append({"string": s, "fret": int(value) - 1, "start": int(start), "end": int(end)})
    notes.sort(key=lambda n: (n["start"], n["string"]))
    return notes


def resolve_duplicate_pitches(
    notes: list[dict],
    proba: np.ndarray,
    hand_weight: float = 0.5,
    min_overlap: float = 0.5,
    window_seconds: float = 2.0,
    open_string_cost: float = 3.0,
    unison_confidence: float | None = None,
) -> list[dict]:
    """Remove one note of every simultaneous same-pitch pair on different strings.

    The model decides each string separately, so it often predicts one pitch on two strings at once
    (e.g. C#4 on B fret 2 and on G fret 6); guitarists rarely do that. For every pair of notes with
    the same pitch on different strings that overlap for at least `min_overlap` of the shorter note,
    the note with the lower score is removed:

        score = log(mean model probability of that string+fret over the note + 0.001)
                - hand_weight * trust * distance to the local hand position

    The hand position is the median fret of the other fretted notes starting within
    +-window_seconds; trust = 1 / (1 + std of those frets / 3), so a spread-out context counts less;
    an open string is `open_string_cost` frets away from any hand position. Notes on the same string
    (repeated notes) and notes of different pitches are never compared. A note that already caused a
    removal is never removed itself, so every removed note has a same-pitch partner that is kept.
    If `unison_confidence` is set, pairs where both notes have a mean probability >= that value are
    kept (a played unison). Defaults are the values tuned on the validation guitarist.

    notes: output of frames_to_notes for one recording; proba: (n_frames, 6, 21) model probabilities
    of the same recording. Returns the kept notes in their original order.
    """
    n = len(notes)
    if n < 2:
        return list(notes)
    string = np.array([x["string"] for x in notes])
    fret = np.array([x["fret"] for x in notes])
    start = np.array([x["start"] for x in notes])
    end = np.array([x["end"] for x in notes])
    pitch = OPEN_STRING_MIDI[string] + fret
    conf = np.array([proba[start[i]:end[i], string[i], fret[i] + 1].mean() for i in range(n)])
    window = int(round(window_seconds / frame_seconds()))
    alive = np.ones(n, dtype=bool)
    protected = np.zeros(n, dtype=bool)
    idx = np.arange(n)

    def hand_cost(i: int) -> float:
        nb = idx[(np.abs(start - start[i]) <= window) & (idx != i) & alive & (fret > 0)]
        if len(nb) == 0:
            return 0.0
        trust = 1.0 / (1.0 + fret[nb].std() / 3.0)
        distance = open_string_cost if fret[i] == 0 else abs(fret[i] - float(np.median(fret[nb])))
        return trust * distance

    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            if pitch[i] != pitch[j] or string[i] == string[j]:
                continue
            overlap = min(end[i], end[j]) - max(start[i], start[j])
            if overlap > 0 and overlap >= min_overlap * min(end[i] - start[i], end[j] - start[j]):
                pairs.append((overlap, i, j))
    for _, i, j in sorted(pairs, reverse=True):  # largest overlap first
        if not (alive[i] and alive[j]):
            continue
        if unison_confidence is not None and min(conf[i], conf[j]) >= unison_confidence:
            continue
        score_i = np.log(conf[i] + 1e-3) - hand_weight * hand_cost(i)
        score_j = np.log(conf[j] + 1e-3) - hand_weight * hand_cost(j)
        winner, loser = (i, j) if score_i >= score_j else (j, i)
        if protected[loser]:
            continue
        alive[loser] = False
        protected[winner] = True
    return [note for note, keep in zip(notes, alive) if keep]


def notes_to_frames(notes: list[dict], n_frames: int) -> np.ndarray:
    """Inverse of frames_to_notes: notes -> (n_frames, 6) classes."""
    classes = np.zeros((n_frames, N_STRINGS), dtype=np.int8)
    for n in notes:
        classes[n["start"]:n["end"], n["string"]] = n["fret"] + 1
    return classes


def clean_frames(classes: np.ndarray, min_frames: int, max_gap: int) -> np.ndarray:
    """Frame predictions after the note clean-up, for frame-level evaluation."""
    return notes_to_frames(frames_to_notes(classes, min_frames, max_gap), len(classes))


def note_metrics(true_notes: list[dict], pred_notes: list[dict], onset_tolerance: int = 2) -> dict:
    """Note-level precision / recall / F1.

    A predicted note is correct when a true note on the same string with the same fret starts within
    onset_tolerance frames (2 frames = 46 ms, close to the usual 50 ms in music transcription). Each
    true note can be matched only once. Returns counts so results can be summed over recordings.
    """
    unmatched = {}  # (string, fret) -> onsets of true notes not matched yet
    for t in true_notes:
        unmatched.setdefault((t["string"], t["fret"]), []).append(t["start"])
    tp = 0
    for p in pred_notes:
        onsets = unmatched.get((p["string"], p["fret"]))
        if not onsets:
            continue
        i = min(range(len(onsets)), key=lambda k: abs(onsets[k] - p["start"]))  # closest true onset
        if abs(onsets[i] - p["start"]) <= onset_tolerance:
            onsets.pop(i)
            tp += 1
    return {"tp": tp, "n_pred": len(pred_notes), "n_true": len(true_notes)}


def prf(tp: int, n_pred: int, n_true: int) -> dict:
    precision = tp / n_pred if n_pred else 0.0
    recall = tp / n_true if n_true else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"note_precision": precision, "note_recall": recall, "note_f1": f1}


def note_name(string: int, fret: int) -> str:
    midi = int(OPEN_STRING_MIDI[string]) + fret
    return f"{NOTE_NAMES[midi % 12]}{midi // 12 - 1}"


def notes_to_json(notes: list[dict]) -> list[dict]:
    """Notes with times in seconds and readable names, for the API response."""
    fs = frame_seconds()
    return [
        {
            "string": STRING_NAMES[n["string"]],
            "string_number": 6 - n["string"],  # guitarists number strings 1 (high e) .. 6 (low E)
            "fret": n["fret"],
            "note": note_name(n["string"], n["fret"]),
            "start": round(n["start"] * fs, 3),
            "end": round(n["end"] * fs, 3),
        }
        for n in notes
    ]


def render_tab(notes: list[dict], line_width: int = 76, chord_tolerance: int = 2) -> str:
    """Notes -> ASCII tab.

    Notes that start within chord_tolerance frames of each other are written in the same column
    (a chord). Longer pauses between notes get more dashes (one extra per ~0.25 s, at most 4), so the
    rhythm is roughly visible. Each block of lines starts with its time in seconds.
    """
    if not notes:
        return "(no notes detected)\n"

    # 1. Group notes into columns.
    columns = []  # each: {"start": frame, "frets": {string: fret}}
    for n in sorted(notes, key=lambda n: (n["start"], n["string"])):
        col = columns[-1] if columns else None
        if col and n["start"] - col["start"] <= chord_tolerance and n["string"] not in col["frets"]:
            col["frets"][n["string"]] = n["fret"]
        else:
            columns.append({"start": n["start"], "frets": {n["string"]: n["fret"]}})

    # 2. Turn each column into 6 equal-width text cells, preceded by dashes for the pause.
    quarter_second = 0.25 / frame_seconds()
    cells = []  # (start frame, [6 strings of equal width], low E first)
    previous_start = None
    for col in columns:
        width = max(len(str(f)) for f in col["frets"].values())
        pause = 1 if previous_start is None else 1 + min(4, int((col["start"] - previous_start) / quarter_second))
        text = ["-" * pause + str(col["frets"].get(s, "")).rjust(width, "-") for s in range(N_STRINGS)]
        cells.append((col["start"], text))
        previous_start = col["start"]

    # 3. Wrap into blocks of at most line_width characters.
    label_width = 2  # "e|"
    blocks, current, current_len = [], [], 0
    for cell in cells:
        cell_len = len(cell[1][0])
        if current and current_len + cell_len + 2 > line_width - label_width:
            blocks.append(current)
            current, current_len = [], 0
        current.append(cell)
        current_len += cell_len

    if current:
        blocks.append(current)

    fs = frame_seconds()
    lines = []
    for block in blocks:
        lines.append(f"[{block[0][0] * fs:.1f}s]")
        for s in reversed(range(N_STRINGS)):  # high e on top
            lines.append(f"{STRING_NAMES[s]}|" + "".join(cell[1][s] for cell in block) + "--|")
        lines.append("")
    return "\n".join(lines)
