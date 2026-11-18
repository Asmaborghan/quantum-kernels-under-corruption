"""Project folders. Every notebook and module imports its paths from here."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"              # original downloads, never edited
PROCESSED = ROOT / "data" / "processed"  # the 16 *_ready.csv files (rebuilt by 01_data_preparation)
RESULTS = ROOT / "results"               # experiment outputs

for _folder in (RAW, PROCESSED, RESULTS):
    _folder.mkdir(parents=True, exist_ok=True)
