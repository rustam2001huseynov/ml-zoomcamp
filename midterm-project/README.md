# Guitar tab transcriber: audio in, guitar tablature out

ML Zoomcamp 2026 midterm project. Upload a recording of a solo acoustic guitar, and a web service
returns the notes as guitar **tablature**: which string, which fret, and when.

```
$ curl -F "file=@production/examples/rock_solo.wav" "http://localhost:9696/transcribe?format=text"
[1.0s]
e|---------------------------------------------------------------------|
B|-----------------------------------------5-4-----2-------------------|
G|----------------2-3-6---6----3--6--6---------6---6----3--6-6--6-8-6--|
D|----3--6--6--3-------------------------------------------------------|
A|-4-------------------------------------------------------------------|
E|---------------------------------------------------------------------|
```

**Try it live:** <https://guitar-tab-xeh9.onrender.com> (free hosting: if it was asleep, the first request takes about a minute).

## Problem

Guitarists learn songs from **tabs**: six lines for the six strings, with fret numbers on them.
Writing one by ear is slow. A transcriber has to solve two things:
1. **Which pitches are sounding?** Up to six notes at once, with overlapping overtones.
2. **On which string is each pitch played?** Most pitches can be played in 2–5 places on the neck
   (95% of the notes in our data), and a tab must pick one.

**As an ML problem:** the audio is cut into 23 ms frames. For every frame and each of the 6
strings, a classifier predicts one of 21 classes: silent, open string, or fret 1–19. Frames are then
joined into notes and written out as a tab. Uses: a first draft of a tab, practice feedback,
searching recordings by what was played.

## Data

[GuitarSet](https://zenodo.org/records/3371780) (Xi et al., ISMIR 2018, CC BY 4.0):
- 360 recordings, 3 hours, of 6 guitarists in 5 styles;
- `.jams` annotations with the exact string, pitch, start and duration of all 62,476 notes,
  recorded with a separate pickup per string.

The data is not in the repository. `data/get_data.py` downloads the audio and annotations (~700 MB),
checks the checksums and computes the features and labels into `data/processed/`. The two 15 s
clips in `production/examples/` are cut from GuitarSet recordings of the test guitarist.

## Approach

- **Features:** Constant-Q spectrogram (168 bins from E2, 2 per semitone, in dB) plus the change
  since the previous frame: 336 features per frame.
- **Validation:** split **by guitarist**, so every score is on a player the model has never heard:
  train = guitarists 1, 2, 3, 5; validation = 0; test = 4. Neighbouring frames are nearly identical,
  so a random split would leak and inflate the scores.
- **Models** (one classifier per string): dummy, logistic regression, decision tree, random forest,
  XGBoost, each tuned on the validation guitarist (C, class weights, depth, leaf size, trees, eta,
  min_child_weight, decision threshold).
- **Evaluation:** accuracy, confusion table, precision, recall, F1, ROC AUC, and 5-fold
  cross-validation with guitarists as folds. Main metric: **tab F1**, where a prediction counts only
  if both string and fret are right.
- **Final model:** XGBoost retrained on 5 guitarists, scored once on the test guitarist. The
  decision threshold and the shortest note to keep were tuned on validation for note-level F1.

## Results

Best version of each model (validation guitarist; CV = mean over 5 guitarist folds). All metrics,
confusion tables and per-string scores are in `research/model_comparison.xlsx`.

| Model | Accuracy | ROC AUC | Tab F1 | CV tab F1 |
|---|---|---|---|---|
| Dummy (always silent) | 0.606 | 0.500 | 0.000 | 0.000 |
| Decision tree | 0.639 | 0.782 | 0.414 | 0.467 |
| Random forest | 0.674 | 0.855 | 0.512 | 0.568 |
| Logistic regression | 0.737 | 0.868 | 0.586 | 0.624 |
| **XGBoost** | **0.752** | **0.893** | **0.610** | **0.656** |

XGBoost wins on all 5 folds. **On the test guitarist** the final model reaches tab F1 **0.719**
(frames) and note F1 **0.569** (whole notes, start within 46 ms). For comparison, TabCNN, a CNN
baseline on GuitarSet, reaches about 0.75 tab F1
([DAFx 2024](https://dafx.de/paper-archive/2024/papers/DAFx24_paper_99.pdf)).

**Limitations:** most errors are timing (a note repeated without a gap looks like one long note;
starts are detected late) and the wrong string for the right pitch. The model was trained on one
acoustic guitar and 6 players, so expect lower accuracy on other guitars or noisy recordings.

## Project structure

```
data/get_data.py               download GuitarSet, compute features and labels
research/notebook.ipynb        EDA, feature importance, every model with tuning and metrics
research/compare_models.py     trains the best version of every model -> model_comparison.xlsx
research/model_comparison.xlsx metrics of every model
production/guitar.py           shared code: features, metrics, per-string model, notes -> tab
production/train.py            trains the final model -> model.bin
production/model.bin           trained model, so training is optional
production/predict.py          FastAPI web service
production/predict_test.py     client that sends an audio file to the service
production/Dockerfile          container for the service
```

## How to run

Run everything from the `midterm-project` folder. Dependencies are pinned in `uv.lock`:

```bash
pip install uv
uv sync                                       # creates .venv with all dependencies
uv run python data/get_data.py                # data, only needed for research and training
uv run jupyter notebook research/notebook.ipynb
uv run python research/compare_models.py      # rebuilds the Excel table, ~45 min
uv run python production/train.py             # retrains production/model.bin, ~15 min
```

For a quick run of the notebook or `compare_models.py` on less data, set `N_TRAIN=5000`.

**Web service:**

```bash
cd production && uv run uvicorn predict:app --host 0.0.0.0 --port 9696
```

**Docker** (no dataset needed):

```bash
docker build -f production/Dockerfile -t guitar-tab .
docker run -it --rm -p 9696:9696 guitar-tab
```

**Test it** (on Windows PowerShell, type `curl.exe`):

```bash
curl http://localhost:9696/health
curl -F "file=@production/examples/rock_solo.wav" "http://localhost:9696/transcribe?format=text"
uv run python production/predict_test.py production/examples/jazz_comp.wav
```

Or open http://localhost:9696 in a browser: `POST /transcribe` → **Try it out** → choose a file →
**Execute**.

| Endpoint | Description |
|---|---|
| `GET /health` | status and settings |
| `POST /transcribe` | form field `file` (WAV, FLAC, OGG or MP3, max 120 s) → JSON with every note (string, fret, pitch, start, end) and the tab |
| `POST /transcribe?format=text` | the tab only |

## Cloud deployment (Render, free)

**Live URL:** <https://guitar-tab-xeh9.onrender.com> (opens the upload form; `GET /health`, `POST /transcribe`). Example:

```bash
curl -F "file=@production/examples/rock_solo.wav" "https://guitar-tab-xeh9.onrender.com/transcribe?format=text"
```

**Deploy it yourself:**

1. Render dashboard → **New → Web Service** → connect this GitHub repository.
2. Language **Docker**, Root Directory `midterm-project`, Dockerfile Path `production/Dockerfile`,
   Instance Type **Free**, Health Check Path `/health`.
3. Test with the commands above, using `https://<your-service>.onrender.com` instead of localhost.

The free instance (512 MB, 0.1 CPU) sleeps after 15 minutes without requests; the first request
then takes about a minute. Tested locally with the same limits: a 15 s clip takes 2.5 s.

## Acknowledgements

GuitarSet: Q. Xi, R. Bittner, J. Pauwels, X. Ye, J. P. Bello, ISMIR 2018, CC BY 4.0.
[ML Zoomcamp](https://github.com/DataTalksClub/machine-learning-zoomcamp) by DataTalks.Club.
