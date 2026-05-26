"""
fetch_data.py — Pull listening data from Spotify API.

Fetches:
  - Top 50 tracks for short_term / medium_term / long_term
  - Audio features for every track (falls back to synthetic if 403)
  - Artist genres via artist lookup
  - Recently played (last 50) for trend analysis

Saves flat CSV to data/raw_tracks.csv.

Usage:
    python fetch_data.py
"""

import os
import time
from datetime import datetime

import pandas as pd

from auth import get_spotify_client

DATA_DIR = "data"
RAW_CSV = os.path.join(DATA_DIR, "raw_tracks.csv")
SYNTHETIC_FLAG = os.path.join(DATA_DIR, ".synthetic_features")  # presence = synthetic mode

TIME_RANGES = ["short_term", "medium_term", "long_term"]
AUDIO_FEATURE_KEYS = [
    "danceability", "energy", "valence", "tempo",
    "acousticness", "instrumentalness", "speechiness",
    "loudness", "liveness",
]


def _log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")


def fetch_top_tracks(sp, time_range: str, limit: int = 50) -> list[dict]:
    """Fetch top tracks for a given time range."""
    _log(f"Fetching top {limit} tracks for {time_range}...")
    results = sp.current_user_top_tracks(limit=limit, time_range=time_range)
    tracks = []
    for item in results["items"]:
        tracks.append({
            "track_id": item["id"],
            "track_name": item["name"],
            "artist_id": item["artists"][0]["id"],
            "artist_name": item["artists"][0]["name"],
            "album_name": item["album"]["name"],
            "popularity": item.get("popularity", 0),
            "duration_ms": item.get("duration_ms", 0),
            "explicit": item.get("explicit", False),
            "time_range": time_range,
            "added_at": None,
        })
    _log(f"  → Got {len(tracks)} tracks for {time_range}")
    return tracks


def fetch_recently_played(sp, limit: int = 50) -> list[dict]:
    """Fetch recently played tracks with timestamps."""
    _log("Fetching recently played tracks...")
    results = sp.current_user_recently_played(limit=limit)
    tracks = []
    for item in results["items"]:
        t = item["track"]
        tracks.append({
            "track_id": t["id"],
            "track_name": t["name"],
            "artist_id": t["artists"][0]["id"],
            "artist_name": t["artists"][0]["name"],
            "album_name": t["album"]["name"],
            "popularity": t.get("popularity", 0),
            "duration_ms": t.get("duration_ms", 0),
            "explicit": t.get("explicit", False),
            "time_range": "recently_played",
            "added_at": item["played_at"],
        })
    _log(f"  → Got {len(tracks)} recently played tracks")
    return tracks


def fetch_audio_features(sp, track_ids: list[str]) -> tuple[dict[str, dict], bool]:
    """
    Try to fetch real audio features. Returns (features_map, is_synthetic).

    Spotify restricted /audio-features for most developer apps in late 2024.
    If we get a 403, we return an empty map and signal the caller to generate
    synthetic features instead so the rest of the pipeline still works.
    """
    import spotipy

    _log(f"Fetching audio features for {len(track_ids)} tracks...")
    features_map = {}
    batch_size = 100

    try:
        for i in range(0, len(track_ids), batch_size):
            batch = track_ids[i : i + batch_size]
            results = sp.audio_features(batch)
            for feat in results:
                if feat is None:
                    continue
                features_map[feat["id"]] = {k: feat.get(k) for k in AUDIO_FEATURE_KEYS}
            time.sleep(0.1)

        _log(f"  → Got real audio features for {len(features_map)} tracks")
        return features_map, False

    except spotipy.exceptions.SpotifyException as e:
        if e.http_status == 403:
            _log("  ⚠  Spotify returned 403 on /audio-features.")
            _log("     This endpoint was restricted for most apps in late 2024.")
            _log("     Switching to SYNTHETIC audio features (genre-seeded + noise).")
            _log("     Clustering still works. Dashboard will show a notice.")
            return {}, True
        raise


# ---------------------------------------------------------------------------
# Genre → rough audio profile seeds (all values normalised to 0-1 except tempo)
# ---------------------------------------------------------------------------
GENRE_SEEDS: dict[str, dict] = {
    "pop":           dict(danceability=0.72, energy=0.68, valence=0.65, tempo=118, acousticness=0.18, instrumentalness=0.02, speechiness=0.06, loudness=0.70, liveness=0.14),
    "hip hop":       dict(danceability=0.80, energy=0.70, valence=0.55, tempo=96,  acousticness=0.12, instrumentalness=0.04, speechiness=0.22, loudness=0.72, liveness=0.12),
    "rap":           dict(danceability=0.78, energy=0.72, valence=0.50, tempo=98,  acousticness=0.10, instrumentalness=0.03, speechiness=0.26, loudness=0.73, liveness=0.13),
    "r&b":           dict(danceability=0.74, energy=0.58, valence=0.58, tempo=100, acousticness=0.25, instrumentalness=0.03, speechiness=0.08, loudness=0.65, liveness=0.11),
    "soul":          dict(danceability=0.66, energy=0.55, valence=0.60, tempo=95,  acousticness=0.40, instrumentalness=0.04, speechiness=0.06, loudness=0.60, liveness=0.16),
    "rock":          dict(danceability=0.54, energy=0.82, valence=0.50, tempo=128, acousticness=0.10, instrumentalness=0.08, speechiness=0.05, loudness=0.82, liveness=0.18),
    "metal":         dict(danceability=0.40, energy=0.92, valence=0.30, tempo=148, acousticness=0.05, instrumentalness=0.20, speechiness=0.05, loudness=0.90, liveness=0.16),
    "indie":         dict(danceability=0.58, energy=0.60, valence=0.52, tempo=118, acousticness=0.28, instrumentalness=0.10, speechiness=0.05, loudness=0.62, liveness=0.14),
    "folk":          dict(danceability=0.48, energy=0.40, valence=0.55, tempo=100, acousticness=0.72, instrumentalness=0.06, speechiness=0.04, loudness=0.48, liveness=0.12),
    "acoustic":      dict(danceability=0.50, energy=0.38, valence=0.54, tempo=96,  acousticness=0.80, instrumentalness=0.05, speechiness=0.04, loudness=0.45, liveness=0.11),
    "jazz":          dict(danceability=0.58, energy=0.42, valence=0.60, tempo=112, acousticness=0.60, instrumentalness=0.30, speechiness=0.04, loudness=0.52, liveness=0.20),
    "classical":     dict(danceability=0.28, energy=0.28, valence=0.42, tempo=108, acousticness=0.90, instrumentalness=0.82, speechiness=0.03, loudness=0.38, liveness=0.10),
    "electronic":    dict(danceability=0.76, energy=0.80, valence=0.55, tempo=128, acousticness=0.05, instrumentalness=0.55, speechiness=0.05, loudness=0.78, liveness=0.10),
    "edm":           dict(danceability=0.80, energy=0.88, valence=0.60, tempo=130, acousticness=0.04, instrumentalness=0.60, speechiness=0.04, loudness=0.82, liveness=0.09),
    "dance":         dict(danceability=0.82, energy=0.82, valence=0.65, tempo=124, acousticness=0.06, instrumentalness=0.25, speechiness=0.05, loudness=0.78, liveness=0.11),
    "latin":         dict(danceability=0.84, energy=0.76, valence=0.74, tempo=110, acousticness=0.22, instrumentalness=0.04, speechiness=0.07, loudness=0.72, liveness=0.15),
    "reggae":        dict(danceability=0.78, energy=0.60, valence=0.72, tempo=90,  acousticness=0.30, instrumentalness=0.06, speechiness=0.08, loudness=0.62, liveness=0.14),
    "country":       dict(danceability=0.60, energy=0.62, valence=0.65, tempo=120, acousticness=0.45, instrumentalness=0.02, speechiness=0.04, loudness=0.65, liveness=0.14),
    "blues":         dict(danceability=0.54, energy=0.52, valence=0.48, tempo=104, acousticness=0.48, instrumentalness=0.10, speechiness=0.04, loudness=0.58, liveness=0.18),
    "punk":          dict(danceability=0.52, energy=0.88, valence=0.52, tempo=158, acousticness=0.05, instrumentalness=0.04, speechiness=0.08, loudness=0.86, liveness=0.22),
    "ambient":       dict(danceability=0.30, energy=0.22, valence=0.35, tempo=88,  acousticness=0.65, instrumentalness=0.75, speechiness=0.03, loudness=0.30, liveness=0.08),
    "k-pop":         dict(danceability=0.78, energy=0.76, valence=0.68, tempo=122, acousticness=0.12, instrumentalness=0.04, speechiness=0.08, loudness=0.74, liveness=0.12),
    "trap":          dict(danceability=0.76, energy=0.68, valence=0.42, tempo=72,  acousticness=0.08, instrumentalness=0.15, speechiness=0.18, loudness=0.75, liveness=0.10),
    "default":       dict(danceability=0.60, energy=0.60, valence=0.55, tempo=112, acousticness=0.25, instrumentalness=0.08, speechiness=0.06, loudness=0.65, liveness=0.13),
}


def _genre_seed(genres: list[str]) -> dict:
    """Pick the best matching genre seed, blending up to 2 matches."""
    matched = []
    for g in genres:
        g_lower = g.lower()
        for key in GENRE_SEEDS:
            if key in g_lower:
                matched.append(GENRE_SEEDS[key])
                break
    if not matched:
        return GENRE_SEEDS["default"]
    if len(matched) == 1:
        return matched[0]
    a, b = matched[0], matched[1]
    return {k: a[k] * 0.6 + b[k] * 0.4 for k in a}


def generate_synthetic_features(
    all_tracks: list[dict],
    genres_map: dict[str, list[str]],
) -> dict[str, dict]:
    """
    Generate plausible audio features seeded by genre + small Gaussian noise.
    Track ID hash ensures the same track always gets the same values.
    """
    import random
    features_map = {}
    for t in all_tracks:
        tid = t["track_id"]
        if tid in features_map:
            continue
        genres = genres_map.get(t["artist_id"], [])
        seed = _genre_seed(genres)
        rng = random.Random(hash(tid))
        noise = lambda: rng.gauss(0, 0.06)

        features_map[tid] = {
            "danceability":     max(0.0, min(1.0, seed["danceability"]     + noise())),
            "energy":           max(0.0, min(1.0, seed["energy"]           + noise())),
            "valence":          max(0.0, min(1.0, seed["valence"]          + noise())),
            "tempo":            max(60,  min(200, seed["tempo"]            + rng.gauss(0, 8))),
            "acousticness":     max(0.0, min(1.0, seed["acousticness"]     + noise())),
            "instrumentalness": max(0.0, min(1.0, seed["instrumentalness"] + noise())),
            "speechiness":      max(0.0, min(1.0, seed["speechiness"]      + noise())),
            "loudness":         max(0.0, min(1.0, seed["loudness"]         + noise())),
            "liveness":         max(0.0, min(1.0, seed["liveness"]         + noise())),
        }
    _log(f"  → Generated synthetic features for {len(features_map)} tracks")
    return features_map


def fetch_artist_genres(sp, artist_ids: list[str]) -> dict[str, list[str]]:
    """
    Fetch genres for a list of artist IDs in batches of 50 (API limit).
    Returns a dict keyed by artist_id -> list of genre strings.
    Falls back to empty dict on 403 (Spotify dev-mode restriction).
    """
    import spotipy

    unique_ids = list(set(artist_ids))
    _log(f"Fetching genres for {len(unique_ids)} unique artists...")
    genres_map = {}
    batch_size = 50

    try:
        for i in range(0, len(unique_ids), batch_size):
            batch = unique_ids[i : i + batch_size]
            results = sp.artists(batch)
            for artist in results["artists"]:
                if artist is None:
                    continue
                genres_map[artist["id"]] = artist.get("genres", [])
            time.sleep(0.1)

        _log(f"  -> Fetched genres for {len(genres_map)} artists")

    except spotipy.exceptions.SpotifyException as e:
        if e.http_status == 403:
            _log("  WARNING: /artists returned 403 (Spotify dev-mode restriction).")
            _log("    Genres will be marked Unknown. Synthetic features will use default seed.")
        else:
            raise

    return genres_map


def build_dataframe(
    all_tracks: list[dict],
    features_map: dict,
    genres_map: dict,
) -> pd.DataFrame:
    """Merge tracks + audio features + genres into one flat DataFrame."""
    rows = []
    for t in all_tracks:
        tid = t["track_id"]
        feat = features_map.get(tid)
        if feat is None:
            continue  # still skip if somehow missing

        genres = genres_map.get(t["artist_id"], [])
        genres_str = ", ".join(genres) if genres else "Unknown"

        row = {
            "track_id": tid,
            "track_name": t["track_name"],
            "artist_name": t["artist_name"],
            "artist_id": t["artist_id"],
            "album_name": t["album_name"],
            "popularity": t.get("popularity", 0),
            "duration_ms": t.get("duration_ms", 0),
            "explicit": t.get("explicit", False),
            "genres": genres_str,
            "time_range": t["time_range"],
            "added_at": t["added_at"],
        }
        row.update(feat)
        rows.append(row)

    return pd.DataFrame(rows)


def fetch_all() -> pd.DataFrame:
    """Main entry point: authenticate, fetch everything, return merged DataFrame."""
    os.makedirs(DATA_DIR, exist_ok=True)
    sp = get_spotify_client()

    # Collect all tracks
    all_tracks = []
    for tr in TIME_RANGES:
        all_tracks.extend(fetch_top_tracks(sp, tr))
    all_tracks.extend(fetch_recently_played(sp))
    _log(f"Total raw track records collected: {len(all_tracks)}")

    unique_track_ids = list({t["track_id"] for t in all_tracks})
    unique_artist_ids = [t["artist_id"] for t in all_tracks]

    # Genres first (needed for synthetic fallback too)
    genres_map = fetch_artist_genres(sp, unique_artist_ids)

    # Audio features (real or synthetic)
    features_map, is_synthetic = fetch_audio_features(sp, unique_track_ids)
    if is_synthetic:
        features_map = generate_synthetic_features(all_tracks, genres_map)
        # Write flag file so dashboard can show a notice
        open(SYNTHETIC_FLAG, "w").close()
    else:
        # Remove stale flag if we got real data
        if os.path.exists(SYNTHETIC_FLAG):
            os.remove(SYNTHETIC_FLAG)

    _log("Merging tracks, features, and genres...")
    df = build_dataframe(all_tracks, features_map, genres_map)
    _log(f"Merged DataFrame: {df.shape[0]} rows × {df.shape[1]} columns")

    df.to_csv(RAW_CSV, index=False)
    _log(f"Saved to {RAW_CSV}")
    return df


if __name__ == "__main__":
    print("=" * 50)
    print("  Spotify Data Fetcher")
    print("=" * 50)
    df = fetch_all()
    print("\nSample rows:")
    print(df[["track_name", "artist_name", "time_range", "energy", "valence"]].head(10).to_string())
    print(f"\n✅ Done! {len(df)} track records saved to {RAW_CSV}")