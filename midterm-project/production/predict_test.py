"""Send an audio file to the running service and print the tab.

    uv run python production/predict_test.py                    # examples/rock_solo.wav to localhost
    uv run python production/predict_test.py production/examples/jazz_comp.wav
    uv run python production/predict_test.py --url https://<your-app>/transcribe
"""

import argparse
from pathlib import Path

import requests

parser = argparse.ArgumentParser()
parser.add_argument("audio", nargs="?", default=str(Path(__file__).resolve().parent / "examples" / "rock_solo.wav"))
parser.add_argument("--url", default="http://localhost:9696/transcribe")
args = parser.parse_args()

with open(args.audio, "rb") as f:
    response = requests.post(args.url, files={"file": f}, timeout=300)
response.raise_for_status()
result = response.json()

print(f"{result['duration_seconds']} s of audio, {result['n_notes']} notes\n")
print("first notes:")
for note in result["notes"][:5]:
    print(f"  {note['start']:6.2f}s  string {note['string']}  fret {note['fret']:2d}  ({note['note']})")
print()
print(result["tab"])
