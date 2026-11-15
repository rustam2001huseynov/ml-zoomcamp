"""Web service: audio file in, guitar tab out.

    cd production && uv run uvicorn predict:app --host 0.0.0.0 --port 9696

Endpoints:
    GET  /                             -> redirects to /docs (upload form)
    GET  /health                      -> {"status": "ok", ...}
    POST /transcribe  (form field "file": WAV / FLAC / OGG / MP3)
         ?format=json (default)        -> notes with times + the tab as text
         ?format=text                  -> only the tab, as plain text
Interactive docs with an upload button: http://localhost:9696/docs

Environment variables: MODEL_PATH (default model.bin), MAX_SECONDS (default 120),
RESOLVE_DUPLICATE_PITCHES=1 to enable guitar.resolve_duplicate_pitches (default 0 = baseline output).
"""

import io
import os
from pathlib import Path
from typing import Literal

import joblib
import librosa
import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import PlainTextResponse, RedirectResponse
from pydantic import BaseModel

import guitar as F
import guitar as tab
from guitar import decide

MODEL_PATH = os.environ.get("MODEL_PATH", str(Path(__file__).resolve().parent / "model.bin"))
MAX_SECONDS = float(os.environ.get("MAX_SECONDS", "120"))  # longer clips are rejected (CPU time)
# Optional post-processing (guitar.resolve_duplicate_pitches); off by default = baseline behaviour.
RESOLVE_DUPLICATE_PITCHES = os.environ.get("RESOLVE_DUPLICATE_PITCHES", "0") == "1"

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]

app = FastAPI(title="Guitar tab transcriber",
              description="Upload a recording of a solo acoustic guitar and get a tab back.")


class Note(BaseModel):
    string: str          # E A D G B e (low to high)
    string_number: int   # 6 (low E) .. 1 (high e), as guitarists count
    fret: int            # 0 = open string
    note: str            # e.g. "G3"
    start: float         # seconds
    end: float           # seconds


class Transcription(BaseModel):
    duration_seconds: float
    n_notes: int
    notes: list[Note]
    tab: str


def transcribe_audio(y: np.ndarray) -> dict:
    """Mono audio at F.SAMPLE_RATE -> notes and tab."""
    duration = len(y) / F.SAMPLE_RATE
    if len(y) < F.HOP_LENGTH or np.abs(y).max() < 1e-4:  # empty or silent
        notes = []
    else:
        X = F.add_deltas(F.compute_cqt(y))
        proba = model.predict_proba(X)
        classes = decide(proba, bundle["threshold"])
        notes = tab.frames_to_notes(classes, bundle["min_note_frames"], bundle["max_gap_frames"])
        if RESOLVE_DUPLICATE_PITCHES:
            notes = tab.resolve_duplicate_pitches(notes, proba)
    return {
        "duration_seconds": round(duration, 2),
        "n_notes": len(notes),
        "notes": tab.notes_to_json(notes),
        "tab": tab.render_tab(notes),
    }


# Warm-up: librosa compiles its CQT code on first use (~20 s). Do it now, not on the first request.
transcribe_audio(np.random.default_rng(0).normal(0, 0.1, F.SAMPLE_RATE).astype(np.float32))


def read_audio(data: bytes) -> np.ndarray:
    try:
        info = sf.info(io.BytesIO(data))
    except Exception:
        raise HTTPException(400, "Could not read the file as audio (supported: WAV, FLAC, OGG, MP3).")
    if info.duration > MAX_SECONDS:
        raise HTTPException(413, f"Audio is {info.duration:.0f} s long; the limit is {MAX_SECONDS:.0f} s.")
    y, _ = librosa.load(io.BytesIO(data), sr=F.SAMPLE_RATE, mono=True)
    return y.astype(np.float32)


@app.get("/", include_in_schema=False)
def index():
    # The interactive docs page has an upload form, so it doubles as a minimal UI
    # for anyone who opens the service URL in a browser.
    return RedirectResponse("/docs")


@app.get("/health")
def health():
    return {"status": "ok", "threshold": bundle["threshold"], "min_note_frames": bundle["min_note_frames"],
            "max_gap_frames": bundle["max_gap_frames"], "max_seconds": MAX_SECONDS,
            "resolve_duplicate_pitches": RESOLVE_DUPLICATE_PITCHES}


# A plain `def` (not async): FastAPI runs it in a worker thread, so the CPU-heavy work doesn't block.
@app.post("/transcribe", response_model=Transcription)
def transcribe(file: UploadFile = File(..., description="Audio file: WAV, FLAC, OGG or MP3"),
               format: Literal["json", "text"] = Query("json", description="'text' returns only the tab")):
    result = transcribe_audio(read_audio(file.file.read()))
    if format == "text":
        return PlainTextResponse(result["tab"])
    return result
