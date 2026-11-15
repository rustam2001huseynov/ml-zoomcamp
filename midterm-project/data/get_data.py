"""Download GuitarSet and build the frame-level dataset.

GuitarSet: https://zenodo.org/records/3371780 (CC BY 4.0). We use the mono mic recordings and the
.jams note annotations.

Output in data/processed/:
    frames.npz  X (frames x 168 CQT bins, dB), y (frames x 6 strings), track, time
    tracks.csv  one row per recording
    notes.csv   one row per annotated note

Run from the project folder: uv run python data/get_data.py
"""

import hashlib
import json
import re
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "production"))
import guitar as G  # noqa: E402

RAW = Path(__file__).resolve().parent / "raw"
OUT = Path(__file__).resolve().parent / "processed"
ZENODO = "https://zenodo.org/records/3371780/files"

# size and md5 from the Zenodo page
ARCHIVES = {
    "annotation.zip": (39132574, "b39b78e63d3446f2e54ddb7a54df9b10"),
    "audio_mono-mic.zip": (656927981, "275966d6610ac34999b58426beb119c3"),
}

# e.g. 00_BN1-129-Eb_comp: guitarist 00, Bossa Nova, progression 1, 129 bpm, key Eb, accompaniment
NAME_RE = re.compile(r"(\d{2})_([A-Za-z]+)(\d)-(\d+)-([A-G][b#]?)_(comp|solo)")
STYLES = {"BN": "Bossa Nova", "Funk": "Funk", "Jazz": "Jazz", "Rock": "Rock", "SS": "Singer-Songwriter"}


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def download(name, size, tries=20):
    part = RAW / (name + ".part")
    for _ in range(tries):
        done = part.stat().st_size if part.exists() else 0
        if done >= size:
            break
        # Zenodo sometimes drops the connection on the big file, so we resume instead of starting over
        request = urllib.request.Request(f"{ZENODO}/{name}?download=1", headers={"Range": f"bytes={done}-"})
        try:
            with urllib.request.urlopen(request, timeout=60) as r:
                if r.status != 206:
                    done = 0
                with open(part, "ab" if done else "wb") as f:
                    while chunk := r.read(1 << 20):
                        f.write(chunk)
                        done += len(chunk)
                        print(f"\r{name}: {done / 1e6:.0f} / {size / 1e6:.0f} MB", end="", flush=True)
            print()
        except OSError as e:
            print(f"\n{name}: {e}, retrying")
            time.sleep(5)
    part.replace(RAW / name)


def extract(name):
    folder = RAW / name.removesuffix(".zip")
    folder.mkdir(exist_ok=True)
    with zipfile.ZipFile(RAW / name) as z:
        for item in z.infolist():
            # we want a flat folder of files, without the macOS junk some archives carry
            if item.is_dir() or item.filename.startswith("__MACOSX"):
                continue
            (folder / Path(item.filename).name).write_bytes(z.read(item))


def get_raw():
    RAW.mkdir(parents=True, exist_ok=True)
    for name, (size, checksum) in ARCHIVES.items():
        folder = RAW / name.removesuffix(".zip")
        if folder.exists() and any(folder.iterdir()):
            print(f"{name}: already there")
            continue
        archive = RAW / name
        if not (archive.exists() and md5(archive) == checksum):
            archive.unlink(missing_ok=True)
            download(name, size)
        if md5(archive) != checksum:
            sys.exit(f"{name}: checksum mismatch, delete it and run again")
        extract(name)
        print(f"{name}: extracted to {folder}")


def read_notes(path):
    jam = json.loads(path.read_text())
    rows = []
    for ann in jam["annotations"]:
        if ann["namespace"] != "note_midi":  # there is one of these per string
            continue
        string = int(ann["annotation_metadata"]["data_source"])  # 0 = low E
        for n in ann["data"]:
            rows.append({
                "string": string,
                "midi": n["value"],
                "fret": round(n["value"]) - int(G.OPEN_STRING_MIDI[string]),
                "start": n["time"],
                "duration": n["duration"],
            })
    return pd.DataFrame(rows)


def frame_labels(notes, n_frames):
    y = np.zeros((n_frames, G.N_STRINGS), dtype=np.int8)
    fs = G.frame_seconds()
    for n in notes.itertuples():
        start, end = int(np.ceil(n.start / fs)), int(np.ceil((n.start + n.duration) / fs))
        y[start:end, n.string] = G.fret_to_class(n.fret)
    return y


def process(name):
    player, style, progression, tempo, key, mode = NAME_RE.fullmatch(name).groups()
    audio = G.load_audio(RAW / "audio_mono-mic" / f"{name}_mic.wav")
    X = G.compute_cqt(audio)
    notes = read_notes(RAW / "annotation" / f"{name}.jams")
    info = {
        "name": name, "player": int(player), "style": STYLES[style], "progression": int(progression),
        "tempo": int(tempo), "key": key, "mode": mode,
        "duration": len(audio) / G.SAMPLE_RATE, "n_frames": len(X), "n_notes": len(notes),
    }
    y = frame_labels(notes, len(X))
    notes.insert(0, "name", name)
    return info, X.astype(np.float16), y, notes


def prepare():
    names = sorted(p.stem for p in (RAW / "annotation").glob("*.jams"))
    print(f"computing features for {len(names)} recordings...")
    infos, X, y, notes = zip(*Parallel(n_jobs=-1)(delayed(process)(name) for name in names))

    OUT.mkdir(exist_ok=True)
    np.savez(
        OUT / "frames.npz",
        X=np.concatenate(X),
        y=np.concatenate(y),
        track=np.concatenate([np.full(len(x), i, dtype=np.int16) for i, x in enumerate(X)]),
        time=np.concatenate([np.arange(len(x), dtype=np.float32) * G.frame_seconds() for x in X]),
    )
    tracks = pd.DataFrame(infos).rename_axis("track_id")
    tracks.to_csv(OUT / "tracks.csv")
    for i, n in enumerate(notes):
        n.insert(0, "track_id", i)
    pd.concat(notes, ignore_index=True).to_csv(OUT / "notes.csv", index=False)
    print(f"saved {tracks['n_frames'].sum()} frames, {tracks['n_notes'].sum()} notes, "
          f"{tracks['duration'].sum() / 3600:.2f} h of audio")


if __name__ == "__main__":
    get_raw()
    prepare()
