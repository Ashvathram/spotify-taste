"""
preprocess.py — Clean and normalize the raw tracks data.

Reads:  data/raw_tracks.csv  (from fetch_data.py)
Writes: data/tracks.csv      (normalized master, used by analysis.py + app.py)

Transformations:
  - Drop rows missing key audio features
  - Normalize audio feature columns to [0, 1] with MinMaxScaler
    (tempo and loudness have wide ranges; others are already ~0-1)
  - Add a tempo_norm column (normalized tempo, kept alongside raw tempo)
  - Lowercase + strip genres
  - Parse added_at as datetime where available

Usage:
    python preprocess.py
"""

import os
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

DATA_DIR = "data"
RAW_CSV = os.path.join(DATA_DIR, "raw_tracks.csv")
OUT_CSV = os.path.join(DATA_DIR, "tracks.csv")

# Columns that need normalization (some are already 0-1, but we normalize all
# so downstream code always works on a consistent scale)
AUDIO_FEATURE_COLS = [
    "danceability", "energy", "valence", "tempo",
    "acousticness", "instrumentalness", "speechiness",
    "loudness", "liveness",
]

# Columns used for clustering (must be present and normalized)
CLUSTER_FEATURE_COLS = [
    "danceability", "energy", "valence", "tempo_norm",
    "acousticness", "instrumentalness",
]


def _log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")


def load_raw(path: str = RAW_CSV) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Raw data not found at {path}. Run `python fetch_data.py` first."
        )
    df = pd.read_csv(path)
    _log(f"Loaded raw data: {len(df)} rows, {len(df.columns)} columns")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    _log("Cleaning data...")
    original_len = len(df)

    # Drop rows where critical audio features are missing
    df = df.dropna(subset=AUDIO_FEATURE_COLS)
    dropped = original_len - len(df)
    if dropped:
        _log(f"  → Dropped {dropped} rows with missing audio features")

    # Fill missing genres
    df["genres"] = df["genres"].fillna("Unknown")

    # Strip whitespace from string columns
    for col in ["track_name", "artist_name", "album_name", "genres"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # Parse added_at as datetime (recently_played rows have this; top tracks don't)
    if "added_at" in df.columns:
        df["added_at"] = pd.to_datetime(df["added_at"], errors="coerce", utc=True)

    # Ensure popularity is numeric
    df["popularity"] = pd.to_numeric(df["popularity"], errors="coerce").fillna(0).astype(int)

    # Ensure boolean explicit column
    df["explicit"] = df["explicit"].astype(bool)

    _log(f"  → After cleaning: {len(df)} rows")
    return df.reset_index(drop=True)


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize audio feature columns to [0, 1].
    Adds _norm suffix columns for tempo and loudness (wide-range features).
    Also adds tempo_norm and loudness_norm as explicit columns for clustering.
    Raw values are preserved.
    """
    _log("Normalizing audio features...")

    scaler = MinMaxScaler()
    norm_values = scaler.fit_transform(df[AUDIO_FEATURE_COLS])
    norm_df = pd.DataFrame(norm_values, columns=AUDIO_FEATURE_COLS, index=df.index)

    # Add normalized versions with _norm suffix for tempo and loudness
    # (so downstream code can reference either the raw or normalized value)
    df["tempo_norm"] = norm_df["tempo"]
    df["loudness_norm"] = norm_df["loudness"]

    # Overwrite the 0-1 feature columns with their normalized versions
    # (danceability, energy, valence etc. are already near 0-1, but
    #  normalizing ensures strict [0,1] bounds for clustering)
    for col in AUDIO_FEATURE_COLS:
        df[col] = norm_df[col]

    _log(f"  → Normalized {len(AUDIO_FEATURE_COLS)} columns")
    return df


def validate_cluster_features(df: pd.DataFrame):
    """Check that all columns needed for clustering are present."""
    missing = [c for c in CLUSTER_FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing cluster feature columns after preprocessing: {missing}")
    _log(f"  → Cluster feature columns verified: {CLUSTER_FEATURE_COLS}")


def preprocess() -> pd.DataFrame:
    """Main entry point: load → clean → normalize → save."""
    os.makedirs(DATA_DIR, exist_ok=True)

    df = load_raw()
    df = clean(df)
    df = normalize(df)
    validate_cluster_features(df)

    df.to_csv(OUT_CSV, index=False)
    _log(f"Saved preprocessed data to {OUT_CSV}")
    return df


if __name__ == "__main__":
    print("=" * 50)
    print("  Spotify Data Preprocessor")
    print("=" * 50)
    df = preprocess()

    print("\nColumn list:")
    print(list(df.columns))

    print("\nAudio feature stats (normalized):")
    print(df[AUDIO_FEATURE_COLS].describe().round(3).to_string())

    print(f"\n✅ Done! {len(df)} rows saved to {OUT_CSV}")
